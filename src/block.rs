//! The [`Generator`] trait and [`BlockRng`]
//!
//! Trait [`Generator`] and marker trait [`CryptoGenerator`] may be implemented
//! by block-generators; that is PRNGs whose output is a *block* of words, such
//! as `[u32; 16]`.
//!
//! The struct [`BlockRng`] wraps such a [`Generator`] together with an output
//! buffer and implements several methods (e.g. [`BlockRng::next_word`]) to
//! assist in the implementation of [`RngCore`]. Note that (unlike in earlier
//! versions of `rand_core`) [`BlockRng`] itself does not implement [`RngCore`]
//! since in practice we found it was always beneficial to use a wrapper type
//! over [`BlockRng`].
//!
//! # Example
//!
//! ```
//! use rand_core::{RngCore, SeedableRng};
//! use rand_core::block::{Generator, BlockRng};
//!
//! struct MyRngCore {
//!     // Generator state ...
//! #    state: [u32; 8],
//! }
//!
//! impl Generator for MyRngCore {
//!     type Output = [u32; 8];
//!
//!     fn generate(&mut self, output: &mut Self::Output) {
//!         // Write a new block to output...
//! #        *output = self.state;
//!     }
//! }
//!
//! // Our RNG is a wrapper over BlockRng
//! pub struct MyRng(BlockRng<MyRngCore>);
//!
//! impl SeedableRng for MyRng {
//!     type Seed = [u8; 32];
//!     fn from_seed(seed: Self::Seed) -> Self {
//!         let core = MyRngCore {
//!             // ...
//! #            state: {
//! #                let mut buf = [0u32; 8];
//! #                rand_core::le::read_u32_into(&seed, &mut buf);
//! #                buf
//! #            }
//!         };
//!         MyRng(BlockRng::new(core))
//!     }
//! }
//!
//! impl RngCore for MyRng {
//!     #[inline]
//!     fn next_u32(&mut self) -> u32 {
//!         self.0.next_word()
//!     }
//!
//!     #[inline]
//!     fn next_u64(&mut self) -> u64 {
//!         self.0.next_u64_from_u32()
//!     }
//!
//!     #[inline]
//!     fn fill_bytes(&mut self, bytes: &mut [u8]) {
//!         self.0.fill_bytes(bytes)
//!     }
//! }
//!
//! // And if applicable: impl CryptoRng for MyRng {}
//!
//! let mut rng = MyRng::seed_from_u64(0);
//! println!("First value: {}", rng.next_u32());
//! # assert_eq!(rng.next_u32(), 1171109249);
//! ```
//!
//! # ReseedingRng
//!
//! The [`Generator`] trait supports usage of [`rand::rngs::ReseedingRng`].
//! This requires that [`SeedableRng`] be implemented on the "core" generator.
//! Additionally, it may be useful to implement [`CryptoGenerator`].
//! (This is in addition to any implementations on an [`RngCore`] type.)
//!
//! [`Generator`]: crate::block::Generator
//! [`RngCore`]: crate::RngCore
//! [`SeedableRng`]: crate::SeedableRng
//! [`rand::rngs::ReseedingRng`]: https://docs.rs/rand/latest/rand/rngs/struct.ReseedingRng.html

use crate::le::{Word, fill_via_chunks};
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
/// See [`CryptoRng`](crate::CryptoRng) docs for more information.
pub trait CryptoGenerator: Generator {}

/// RNG functionality for a block [`Generator`]
///
/// This type encompasses a [`Generator`] [`core`](Self::core) and a buffer.
/// It provides optimized implementations of methods required by an [`RngCore`].
///
/// All values are consumed in-order of generation. No whole words (e.g. `u32`
/// or `u64`) are discarded, though where a word is partially used (e.g. for a
/// byte-fill whose length is not a multiple of the word size) the rest of the
/// word is discarded.
///
/// [`RngCore`]: crate::RngCore
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

impl<W: Copy + Default, const N: usize, G: Generator<Output = [W; N]>> BlockRng<G> {
    /// Create a new `BlockRng` from an existing RNG implementing
    /// `Generator`. Results will be generated on first use.
    #[inline]
    pub fn new(core: G) -> BlockRng<G> {
        BlockRng {
            core,
            index: N,
            results: [W::default(); N],
        }
    }
}

impl<W: Clone, const N: usize, G: Generator<Output = [W; N]>> BlockRng<G> {
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

    /// Generate the next word (e.g. `u32`)
    #[inline]
    pub fn next_word(&mut self) -> W {
        if self.index >= N {
            self.generate_and_set(0);
        }

        let value = self.results[self.index].clone();
        self.index += 1;
        value
    }
}

impl<const N: usize, G: Generator<Output = [u32; N]>> BlockRng<G> {
    /// Generate a `u64` from two `u32` words
    #[inline]
    pub fn next_u64_from_u32(&mut self) -> u64 {
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
}

impl<W: Word, const N: usize, G: Generator<Output = [W; N]>> BlockRng<G> {
    /// Fill `dest`
    #[inline]
    pub fn fill_bytes(&mut self, dest: &mut [u8]) {
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

#[cfg(test)]
mod test {
    use crate::SeedableRng;
    use crate::block::{BlockRng, Generator};

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
        let mut rng1 = BlockRng::new(DummyRng::from_seed([1, 2, 3, 4]));
        let mut rng2 = rng1.clone();
        let mut rng3 = rng1.clone();

        let mut a = [0; 16];
        a[..4].copy_from_slice(&rng1.next_word().to_le_bytes());
        a[4..12].copy_from_slice(&rng1.next_u64_from_u32().to_le_bytes());
        a[12..].copy_from_slice(&rng1.next_word().to_le_bytes());

        let mut b = [0; 16];
        b[..4].copy_from_slice(&rng2.next_word().to_le_bytes());
        b[4..8].copy_from_slice(&rng2.next_word().to_le_bytes());
        b[8..].copy_from_slice(&rng2.next_u64_from_u32().to_le_bytes());
        assert_eq!(a, b);

        let mut c = [0; 16];
        c[..8].copy_from_slice(&rng3.next_u64_from_u32().to_le_bytes());
        c[8..12].copy_from_slice(&rng3.next_word().to_le_bytes());
        c[12..].copy_from_slice(&rng3.next_word().to_le_bytes());
        assert_eq!(a, c);
    }

    #[test]
    fn blockrng_next_u64() {
        let mut rng = BlockRng::new(DummyRng::from_seed([1, 2, 3, 4]));
        let result_size = rng.results.len();
        for _i in 0..result_size / 2 - 1 {
            rng.next_u64_from_u32();
        }
        rng.next_word();

        let _ = rng.next_u64_from_u32();
        assert_eq!(rng.index(), 1);
    }
}
