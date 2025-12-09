//! The `Generator` trait and implementation helpers
//!
//! The [`Generator`] trait exists to assist in the implementation of RNGs
//! which generate a block of data in a cache instead of returning generated
//! values directly.
//!
//! Usage of this trait is optional, but provides two advantages:
//! implementations only need to concern themselves with generation of the
//! block, not the various [`RngCore`] methods (especially [`fill_bytes`], where
//! the optimal implementations are not trivial), and this allows
//! `ReseedingRng` (see [`rand`](https://docs.rs/rand) crate) perform periodic
//! reseeding with very low overhead.
//!
//! # Example
//!
//! ```no_run
//! use rand_core::{RngCore, SeedableRng};
//! use rand_core::block::{Generator, BlockRng};
//!
//! struct MyRngCore;
//!
//! impl Generator for MyRngCore {
//!     type Output = [u32; 16];
//!
//!     fn generate(&mut self, output: &mut Self::Output) {
//!         unimplemented!()
//!     }
//! }
//!
//! impl SeedableRng for MyRngCore {
//!     type Seed = [u8; 32];
//!     fn from_seed(seed: Self::Seed) -> Self {
//!         unimplemented!()
//!     }
//! }
//!
//! // optionally, also implement CryptoGenerator for MyRngCore
//!
//! // Final RNG.
//! let mut rng = BlockRng::<MyRngCore>::seed_from_u64(0);
//! println!("First value: {}", rng.next_u32());
//! ```
//!
//! [`Generator`]: crate::block::Generator
//! [`fill_bytes`]: RngCore::fill_bytes

use crate::le::fill_via_chunks;
use crate::{CryptoRng, RngCore, SeedableRng, TryRngCore};
use core::fmt;

/// A random (block) generator
pub trait Generator {
    /// The output type.
    ///
    /// For use with [`rand_core::block`](crate::block) code this must be `[u32; _]` or `[u64; _]`.
    type Output;

    /// Generate a new block of `output`.
    ///
    /// This must fill `output` with random data.
    fn generate(&mut self, output: &mut Self::Output);

    /// Destruct the output buffer
    ///
    /// This method is called on [`Drop`] of the [`Self::Output`] buffer.
    /// The default implementation does nothing.
    #[inline]
    fn drop(&mut self, output: &mut Self::Output) {
        let _ = output;
    }
}

/// A cryptographically secure generator
///
/// This is a marker trait used to indicate that a [`Generator`] implementation
/// is supposed to be cryptographically secure.
///
/// Mock generators should not implement this trait *except* under a
/// `#[cfg(test)]` attribute to ensure that mock "crypto" generators cannot be
/// used in production.
///
/// See [`CryptoRng`] docs for more information.
pub trait CryptoGenerator: Generator {}

/// A wrapper type implementing [`RngCore`] for some type implementing
/// [`Generator`] with `u32` array buffer; i.e. this can be used to implement
/// a full RNG from just a `generate` function.
///
/// The `core` field may be accessed directly but the results buffer may not.
/// PRNG implementations can simply use a type alias
/// (`pub type MyRng = BlockRng<MyRngCore>;`) but might prefer to use a
/// wrapper type (`pub struct MyRng(BlockRng<MyRngCore>);`); the latter must
/// re-implement `RngCore` but hides the implementation details and allows
/// extra functionality to be defined on the RNG
/// (e.g. `impl MyRng { fn set_stream(...){...} }`).
///
/// `BlockRng` has heavily optimized implementations of the [`RngCore`] methods
/// reading values from the results buffer, as well as
/// calling [`Generator::generate`] directly on the output array when
/// [`fill_bytes`] is called on a large array. These methods also handle
/// the bookkeeping of when to generate a new batch of values.
///
/// No whole generated `u32` values are thrown away and all values are consumed
/// in-order. [`next_u32`] simply takes the next available `u32` value.
/// [`next_u64`] is implemented by combining two `u32` values, least
/// significant first. [`fill_bytes`] consume a whole number of `u32` values,
/// converting each `u32` to a byte slice in little-endian order. If the requested byte
/// length is not a multiple of 4, some bytes will be discarded.
///
/// See also [`BlockRng64`] which uses `u64` array buffers. Currently there is
/// no direct support for other buffer types.
///
/// For easy initialization `BlockRng` also implements [`SeedableRng`].
///
/// [`next_u32`]: RngCore::next_u32
/// [`next_u64`]: RngCore::next_u64
/// [`fill_bytes`]: RngCore::fill_bytes
#[derive(Clone)]
pub struct BlockRng<G: Generator> {
    results: G::Output,
    index: usize,
    /// The *core* part of the RNG, implementing the `generate` function.
    pub core: G,
}

// Custom Debug implementation that does not expose the contents of `results`.
impl<G: Generator + fmt::Debug> fmt::Debug for BlockRng<G> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("BlockRng")
            .field("core", &self.core)
            .field("index", &self.index)
            .finish()
    }
}

impl<G: Generator> Drop for BlockRng<G> {
    fn drop(&mut self) {
        self.core.drop(&mut self.results);
    }
}

impl<const N: usize, G: Generator<Output = [u32; N]>> BlockRng<G> {
    /// Create a new `BlockRng` from an existing RNG implementing
    /// `Generator`. Results will be generated on first use.
    #[inline]
    pub fn new(core: G) -> BlockRng<G> {
        BlockRng {
            core,
            index: N,
            results: [0; N],
        }
    }

    /// Get the index into the result buffer.
    ///
    /// If this is equal to or larger than the size of the result buffer then
    /// the buffer is "empty" and `generate()` must be called to produce new
    /// results.
    #[inline(always)]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Reset the number of available results.
    /// This will force a new set of results to be generated on next use.
    #[inline]
    pub fn reset(&mut self) {
        self.index = N;
    }

    /// Generate a new set of results immediately, setting the index to the
    /// given value.
    #[inline]
    pub fn generate_and_set(&mut self, index: usize) {
        assert!(index < N);
        self.core.generate(&mut self.results);
        self.index = index;
    }
}

impl<const N: usize, G: Generator<Output = [u32; N]>> RngCore for BlockRng<G> {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.generate_and_set(0);
        }

        let value = self.results[self.index];
        self.index += 1;
        value
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let read_u64 = |results: &[u32], index| {
            let data = &results[index..=index + 1];
            (u64::from(data[1]) << 32) | u64::from(data[0])
        };

        let index = self.index;
        if index < N - 1 {
            self.index += 2;
            // Read an u64 from the current index
            read_u64(&self.results, index)
        } else if index >= N {
            self.generate_and_set(2);
            read_u64(&self.results, 0)
        } else {
            let x = u64::from(self.results[N - 1]);
            self.generate_and_set(1);
            let y = u64::from(self.results[0]);
            (y << 32) | x
        }
    }

    #[inline]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut read_len = 0;
        while read_len < dest.len() {
            if self.index >= N {
                self.generate_and_set(0);
            }
            let (consumed_u32, filled_u8) =
                fill_via_chunks(&self.results[self.index..], &mut dest[read_len..]);

            self.index += consumed_u32;
            read_len += filled_u8;
        }
    }
}

impl<const N: usize, G: Generator<Output = [u32; N]> + SeedableRng> SeedableRng for BlockRng<G> {
    type Seed = G::Seed;

    #[inline(always)]
    fn from_seed(seed: Self::Seed) -> Self {
        Self::new(G::from_seed(seed))
    }

    #[inline(always)]
    fn seed_from_u64(seed: u64) -> Self {
        Self::new(G::seed_from_u64(seed))
    }

    #[inline(always)]
    fn from_rng<S: RngCore + ?Sized>(rng: &mut S) -> Self {
        Self::new(G::from_rng(rng))
    }

    #[inline(always)]
    fn try_from_rng<S: TryRngCore + ?Sized>(rng: &mut S) -> Result<Self, S::Error> {
        G::try_from_rng(rng).map(Self::new)
    }
}

impl<const N: usize, G: CryptoGenerator<Output = [u32; N]>> CryptoRng for BlockRng<G> {}

/// A wrapper type implementing [`RngCore`] for some type implementing
/// [`Generator`] with `u64` array buffer; i.e. this can be used to implement
/// a full RNG from just a `generate` function.
///
/// This is similar to [`BlockRng`], but specialized for algorithms that operate
/// on `u64` values.
///
/// No whole generated `u64` values are thrown away and all values are consumed
/// in-order. [`next_u64`] simply takes the next available `u64` value.
/// [`next_u32`] is however a bit special: half of a `u64` is consumed, leaving
/// the other half in the buffer. If the next function called is [`next_u32`]
/// then the other half is then consumed, however both [`next_u64`] and
/// [`fill_bytes`] discard the rest of any half-consumed `u64`s when called.
///
/// [`fill_bytes`] consumes a whole number of `u64` values. If the requested length
/// is not a multiple of 8, some bytes will be discarded.
///
/// [`next_u32`]: RngCore::next_u32
/// [`next_u64`]: RngCore::next_u64
/// [`fill_bytes`]: RngCore::fill_bytes
#[derive(Clone)]
pub struct BlockRng64<G: Generator + ?Sized> {
    results: G::Output,
    index: usize,
    half_used: bool, // true if only half of the previous result is used
    /// The *core* part of the RNG, implementing the `generate` function.
    pub core: G,
}

// Custom Debug implementation that does not expose the contents of `results`.
impl<G: Generator + fmt::Debug> fmt::Debug for BlockRng64<G> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("BlockRng64")
            .field("core", &self.core)
            .field("index", &self.index)
            .field("half_used", &self.half_used)
            .finish()
    }
}

impl<const N: usize, G: Generator<Output = [u64; N]>> BlockRng64<G> {
    /// Create a new `BlockRng` from an existing RNG implementing
    /// `Generator`. Results will be generated on first use.
    #[inline]
    pub fn new(core: G) -> BlockRng64<G> {
        BlockRng64 {
            core,
            index: N,
            half_used: false,
            results: [0; N],
        }
    }

    /// Get the index into the result buffer.
    ///
    /// If this is equal to or larger than the size of the result buffer then
    /// the buffer is "empty" and `generate()` must be called to produce new
    /// results.
    #[inline(always)]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Reset the number of available results.
    /// This will force a new set of results to be generated on next use.
    #[inline]
    pub fn reset(&mut self) {
        self.index = N;
        self.half_used = false;
    }

    /// Generate a new set of results immediately, setting the index to the
    /// given value.
    #[inline]
    pub fn generate_and_set(&mut self, index: usize) {
        assert!(index < N);
        self.core.generate(&mut self.results);
        self.index = index;
        self.half_used = false;
    }
}

impl<const N: usize, G: Generator<Output = [u64; N]>> RngCore for BlockRng64<G> {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        let mut index = self.index - self.half_used as usize;
        if index >= N {
            self.core.generate(&mut self.results);
            self.index = 0;
            index = 0;
            // `self.half_used` is by definition `false`
            self.half_used = false;
        }

        let shift = 32 * (self.half_used as usize);

        self.half_used = !self.half_used;
        self.index += self.half_used as usize;

        (self.results[index] >> shift) as u32
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        if self.index >= N {
            self.core.generate(&mut self.results);
            self.index = 0;
        }

        let value = self.results[self.index];
        self.index += 1;
        self.half_used = false;
        value
    }

    #[inline]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut read_len = 0;
        self.half_used = false;
        while read_len < dest.len() {
            if self.index >= N {
                self.core.generate(&mut self.results);
                self.index = 0;
            }

            let (consumed_u64, filled_u8) =
                fill_via_chunks(&self.results[self.index..], &mut dest[read_len..]);

            self.index += consumed_u64;
            read_len += filled_u8;
        }
    }
}

impl<const N: usize, G: Generator<Output = [u64; N]> + SeedableRng> SeedableRng for BlockRng64<G> {
    type Seed = G::Seed;

    #[inline(always)]
    fn from_seed(seed: Self::Seed) -> Self {
        Self::new(G::from_seed(seed))
    }

    #[inline(always)]
    fn seed_from_u64(seed: u64) -> Self {
        Self::new(G::seed_from_u64(seed))
    }

    #[inline(always)]
    fn from_rng<S: RngCore + ?Sized>(rng: &mut S) -> Self {
        Self::new(G::from_rng(rng))
    }

    #[inline(always)]
    fn try_from_rng<S: TryRngCore + ?Sized>(rng: &mut S) -> Result<Self, S::Error> {
        G::try_from_rng(rng).map(Self::new)
    }
}

impl<const N: usize, G: CryptoGenerator<Output = [u64; N]>> CryptoRng for BlockRng64<G> {}

#[cfg(test)]
mod test {
    use crate::block::{BlockRng, BlockRng64, Generator};
    use crate::{RngCore, SeedableRng};

    #[derive(Debug, Clone)]
    struct DummyRng {
        counter: u32,
    }

    impl Generator for DummyRng {
        type Output = [u32; 16];

        fn generate(&mut self, output: &mut Self::Output) {
            for item in output {
                *item = self.counter;
                self.counter = self.counter.wrapping_add(3511615421);
            }
        }
    }

    impl SeedableRng for DummyRng {
        type Seed = [u8; 4];

        fn from_seed(seed: Self::Seed) -> Self {
            DummyRng {
                counter: u32::from_le_bytes(seed),
            }
        }
    }

    #[test]
    fn blockrng_next_u32_vs_next_u64() {
        let mut rng1 = BlockRng::<DummyRng>::from_seed([1, 2, 3, 4]);
        let mut rng2 = rng1.clone();
        let mut rng3 = rng1.clone();

        let mut a = [0; 16];
        a[..4].copy_from_slice(&rng1.next_u32().to_le_bytes());
        a[4..12].copy_from_slice(&rng1.next_u64().to_le_bytes());
        a[12..].copy_from_slice(&rng1.next_u32().to_le_bytes());

        let mut b = [0; 16];
        b[..4].copy_from_slice(&rng2.next_u32().to_le_bytes());
        b[4..8].copy_from_slice(&rng2.next_u32().to_le_bytes());
        b[8..].copy_from_slice(&rng2.next_u64().to_le_bytes());
        assert_eq!(a, b);

        let mut c = [0; 16];
        c[..8].copy_from_slice(&rng3.next_u64().to_le_bytes());
        c[8..12].copy_from_slice(&rng3.next_u32().to_le_bytes());
        c[12..].copy_from_slice(&rng3.next_u32().to_le_bytes());
        assert_eq!(a, c);
    }

    #[derive(Debug, Clone)]
    struct DummyRng64 {
        counter: u64,
    }

    impl Generator for DummyRng64 {
        type Output = [u64; 8];

        fn generate(&mut self, output: &mut Self::Output) {
            for item in output {
                *item = self.counter;
                self.counter = self.counter.wrapping_add(2781463553396133981);
            }
        }
    }

    impl SeedableRng for DummyRng64 {
        type Seed = [u8; 8];

        fn from_seed(seed: Self::Seed) -> Self {
            DummyRng64 {
                counter: u64::from_le_bytes(seed),
            }
        }
    }

    #[test]
    fn blockrng64_next_u32_vs_next_u64() {
        let mut rng1 = BlockRng64::<DummyRng64>::from_seed([1, 2, 3, 4, 5, 6, 7, 8]);
        let mut rng2 = rng1.clone();
        let mut rng3 = rng1.clone();

        let mut a = [0; 16];
        a[..4].copy_from_slice(&rng1.next_u32().to_le_bytes());
        a[4..12].copy_from_slice(&rng1.next_u64().to_le_bytes());
        a[12..].copy_from_slice(&rng1.next_u32().to_le_bytes());

        let mut b = [0; 16];
        b[..4].copy_from_slice(&rng2.next_u32().to_le_bytes());
        b[4..8].copy_from_slice(&rng2.next_u32().to_le_bytes());
        b[8..].copy_from_slice(&rng2.next_u64().to_le_bytes());
        assert_ne!(a, b);
        assert_eq!(&a[..4], &b[..4]);
        assert_eq!(&a[4..12], &b[8..]);

        let mut c = [0; 16];
        c[..8].copy_from_slice(&rng3.next_u64().to_le_bytes());
        c[8..12].copy_from_slice(&rng3.next_u32().to_le_bytes());
        c[12..].copy_from_slice(&rng3.next_u32().to_le_bytes());
        assert_eq!(b, c);
    }

    #[test]
    fn blockrng64_generate_and_set() {
        let mut rng = BlockRng64::<DummyRng64>::from_seed([1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(rng.index(), rng.results.len());

        rng.generate_and_set(5);
        assert_eq!(rng.index(), 5);
    }

    #[test]
    #[should_panic(expected = "index < N")]
    fn blockrng64_generate_and_set_panic() {
        let mut rng = BlockRng64::<DummyRng64>::from_seed([1, 2, 3, 4, 5, 6, 7, 8]);
        rng.generate_and_set(rng.results.len());
    }

    #[test]
    fn blockrng_next_u64() {
        let mut rng = BlockRng::<DummyRng>::from_seed([1, 2, 3, 4]);
        let result_size = rng.results.len();
        for _i in 0..result_size / 2 - 1 {
            rng.next_u64();
        }
        rng.next_u32();

        let _ = rng.next_u64();
        assert_eq!(rng.index(), 1);
    }
}
