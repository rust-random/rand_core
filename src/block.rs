//! The [`BlockRng`] trait and [`BlockBuffer`]
//!
//! Trait [`BlockRng`] may be implemented by block-generators; that is PRNGs
//! whose output is a *block* of words, such as `[u32; 16]`.
//!
//! The struct [`BlockBuffer`] may be used with a [`BlockRng`] to implement
//! [`TryRng`]. Note that (unlike in earlier versions of `rand_core`)
//! [`BlockBuffer`] itself does not implement [`TryRng`].
//!
//! # Example
//!
//! ```
//! use core::convert::Infallible;
//! use rand_core::{Rng, SeedableRng, TryRng};
//! use rand_core::block::{BlockRng, BlockBuffer};
//!
//! struct MyRngCore {
//!     // Generator state ...
//! #    state: [u32; 8],
//! }
//!
//! impl BlockRng for MyRngCore {
//!     type Output = [u32; 8];
//!
//!     fn generate(&mut self, output: &mut Self::Output) {
//!         // Write a new block to output...
//! #        *output = self.state;
//!     }
//! }
//!
//! // Our RNG is a wrapper over BlockBuffer
//! pub struct MyRng {
//!     core: MyRngCore,
//!     buffer: BlockBuffer<MyRngCore>,
//! }
//!
//! impl SeedableRng for MyRng {
//!     type Seed = [u8; 32];
//!     fn from_seed(seed: Self::Seed) -> Self {
//!         MyRng {
//!             core: MyRngCore {
//!                 // ...
//! #               state: rand_core::utils::read_words(&seed),
//!             },
//!             buffer: BlockBuffer::default(),
//!         }
//!     }
//! }
//!
//! impl TryRng for MyRng {
//!     type Error = Infallible;
//!
//!     #[inline]
//!     fn try_next_u32(&mut self) -> Result<u32, Infallible> {
//!         Ok(self.buffer.next_word(&mut self.core))
//!     }
//!
//!     #[inline]
//!     fn try_next_u64(&mut self) -> Result<u64, Infallible> {
//!         Ok(self.buffer.next_u64_from_u32(&mut self.core))
//!     }
//!
//!     #[inline]
//!     fn try_fill_bytes(&mut self, bytes: &mut [u8]) -> Result<(), Infallible> {
//!         Ok(self.buffer.fill_bytes(&mut self.core, bytes))
//!     }
//! }
//!
//! // And if applicable: impl TryCryptoRng for MyRng {}
//!
//! let mut rng = MyRng::seed_from_u64(0);
//! println!("First value: {}", rng.next_u32());
//! # assert_eq!(rng.next_u32(), 1171109249);
//! ```
//!
//! [`TryRng`]: crate::TryRng
//! [`SeedableRng`]: crate::SeedableRng

use crate::utils::Word;

/// A random (block) generator
pub trait BlockRng {
    /// The output type.
    ///
    /// For use with [`rand_core::block`](crate::block) code this must be `[u32; _]` or `[u64; _]`.
    type Output;

    /// Generate a new block of `output`.
    ///
    /// This must fill `output` with random data.
    fn generate(&mut self, output: &mut Self::Output);
}

/// Buffer providing RNG methods over a [`BlockRng`]
///
/// This type does not encapuslate a [`BlockRng`], but is designed to be used
/// alongside one.
/// It provides optimized implementations of methods required by an [`Rng`].
///
/// All values are consumed in-order of generation. No whole words (e.g. `u32`
/// or `u64`) are discarded, though where a word is partially used (e.g. for a
/// byte-fill whose length is not a multiple of the word size) the rest of the
/// word is discarded.
///
/// [`Rng`]: crate::Rng
#[derive(Clone)]
#[allow(missing_debug_implementations)]
pub struct BlockBuffer<G: BlockRng> {
    results: G::Output,
}

impl<W: Word + Default, const N: usize, G: BlockRng<Output = [W; N]>> Default for BlockBuffer<G> {
    #[inline]
    fn default() -> BlockBuffer<G> {
        let mut results = [W::default(); N];
        results[0] = W::from_usize(N);
        BlockBuffer { results }
    }
}

impl<W: Word + Default, const N: usize, G: BlockRng<Output = [W; N]>> BlockBuffer<G> {
    /// Reconstruct from a core and a remaining-results buffer.
    ///
    /// This may be used to deserialize using a `core` and the output of
    /// [`Self::remaining_results`].
    ///
    /// Returns `None` if `remaining_results` is too long.
    pub fn reconstruct(remaining_results: &[W]) -> Option<Self> {
        let mut results = [W::default(); N];
        if remaining_results.len() < N {
            let index = N - remaining_results.len();
            results[index..].copy_from_slice(remaining_results);
            results[0] = W::from_usize(index);
            Some(BlockBuffer { results })
        } else {
            None
        }
    }
}

impl<W: Word, const N: usize, G: BlockRng<Output = [W; N]>> BlockBuffer<G> {
    /// Get the index into the result buffer.
    ///
    /// If this is equal to or larger than the size of the result buffer then
    /// the buffer is "empty" and `generate()` must be called to produce new
    /// results.
    #[inline(always)]
    fn index(&self) -> usize {
        self.results[0].into_usize()
    }

    #[inline(always)]
    fn set_index(&mut self, index: usize) {
        debug_assert!(0 < index && index <= N);
        self.results[0] = W::from_usize(index);
    }

    /// Re-generate buffer contents, skipping the first `n` words
    ///
    /// Existing buffer contents are discarded. A new set of results is
    /// generated (either immediately or when next required). The first `n`
    /// words are skipped (this may be used to set a specific word position).
    ///
    /// # Panics
    ///
    /// This method will panic if `n >= N` where `N` is the buffer size (in
    /// words).
    #[inline]
    pub fn reset_and_skip(&mut self, core: &mut G, n: usize) {
        if n == 0 {
            self.set_index(N);
            return;
        }

        assert!(n < N);
        core.generate(&mut self.results);
        self.set_index(n);
    }

    /// Get the number of words consumed since the start of the block
    ///
    /// The result is in the range `0..N` where `N` is the buffer size (in
    /// words).
    #[inline]
    pub fn word_offset(&self) -> usize {
        let index = self.index();
        if index >= N { 0 } else { index }
    }

    /// Access the unused part of the results buffer
    ///
    /// The length of the returned slice is guaranteed to be less than the
    /// length of `<Self as BlockRng>::Output` (i.e. less than `N` where
    /// `Output = [W; N]`).
    ///
    /// This is a low-level interface intended for serialization.
    /// Results are not marked as consumed.
    #[inline]
    pub fn remaining_results(&self) -> &[W] {
        let index = self.index();
        &self.results[index..]
    }

    /// Generate the next word (e.g. `u32`)
    #[inline]
    pub fn next_word(&mut self, core: &mut G) -> W {
        let mut index = self.index();
        if index >= N {
            core.generate(&mut self.results);
            index = 0;
        }

        let value = self.results[index];
        self.set_index(index + 1);
        value
    }
}

impl<const N: usize, G: BlockRng<Output = [u32; N]>> BlockBuffer<G> {
    /// Generate a `u64` from two `u32` words
    #[inline]
    pub fn next_u64_from_u32(&mut self, core: &mut G) -> u64 {
        let index = self.index();
        let mut new_index;
        let (mut lo, mut hi);
        if index < N - 1 {
            lo = self.results[index];
            hi = self.results[index + 1];
            new_index = index + 2;
        } else {
            lo = self.results[N - 1];
            core.generate(&mut self.results);
            hi = self.results[0];
            new_index = 1;
            if index >= N {
                lo = hi;
                hi = self.results[1];
                new_index = 2;
            }
        }
        self.set_index(new_index);
        (u64::from(hi) << 32) | u64::from(lo)
    }
}

impl<W: Word, const N: usize, G: BlockRng<Output = [W; N]>> BlockBuffer<G> {
    /// Fill `dest`
    #[inline]
    pub fn fill_bytes(&mut self, core: &mut G, dest: &mut [u8]) {
        let mut read_len = 0;
        let mut index = self.index();
        while read_len < dest.len() {
            if index >= N {
                core.generate(&mut self.results);
                index = 0;
            }

            let size = core::mem::size_of::<W>();
            let mut chunks = dest[read_len..].chunks_exact_mut(size);
            let mut src = self.results[index..].iter();

            let zipped = chunks.by_ref().zip(src.by_ref());
            let num_chunks = zipped.len();
            zipped.for_each(|(chunk, src)| chunk.copy_from_slice(src.to_le_bytes().as_ref()));
            index += num_chunks;
            read_len += num_chunks * size;

            if let Some(src) = src.next() {
                // We have consumed all full chunks of dest, but not src.
                let dest_rem = chunks.into_remainder();
                let n = dest_rem.len();
                if n > 0 {
                    dest_rem.copy_from_slice(&src.to_le_bytes().as_ref()[..n]);
                    index += 1;
                    debug_assert_eq!(read_len + n, dest.len());
                }
                break;
            }
        }
        self.set_index(index);
    }
}
