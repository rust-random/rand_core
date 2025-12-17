//! Helper utilities.
//!
//! For cross-platform reproducibility, Little-Endian order (least-significant
//! part first) has been chosen as the standard for inter-type conversion.
//! For example, [`next_u64_via_u32`] generates two `u32` values `x, y`,
//! then outputs `(y << 32) | x`.
//!
//! Byte-swapping (like the std `to_le` functions) is only needed to convert
//! to/from byte sequences, and since its purpose is reproducibility,
//! non-reproducible sources (e.g. `OsRng`) need not bother with it.
//!
//! # Implementing [`SeedableRng`]
//!
//! In many cases, [`SeedableRng::Seed`] must be converted to `[u32]` or `[u64]`.
//! We provide the [`read_words`] helper function for this. The examples below
//! demonstrate how it can be used in practice.
//!
//! [`SeedableRng`]: crate::SeedableRng
//! [`SeedableRng::Seed`]: crate::SeedableRng::Seed
//!
//! # Implementing [`RngCore`]
//!
//! Usually an implementation of [`RngCore`] will implement one of the three methods
//! over its internal source, while remaining methods are implemented on top of it.
//!
//! Some RNGs instead generate fixed-size blocks of data. In this case the implementations must
//! handle buffering of the generated blocks.
//!
//! If an implementation can generate several blocks simultaneously (e.g. using SIMD), we recommend
//! to treat multiple generated blocks as a single large block (i.e. you should treat
//! `[[u32; N]; M]` as `[u32; N * M]`). If the number of simultaneously generated blocks depends
//! on CPU target features, we recommend to use the largest supported number of blocks
//! for all target features.
//!
//! # Examples
//!
//! The examples below demonstrate how functions in this module can be used to implement
//! [`RngCore`] and [`SeedableRng`] for common RNG algorithm classes.
//!
//! ## RNG outputs `u32`
//!
//! ```
//! use rand_core::{RngCore, SeedableRng, utils};
//!
//! pub struct Step32Rng {
//!     state: u32
//! }
//!
//! impl SeedableRng for Step32Rng {
//!     type Seed = [u8; 4];
//!
//!     #[inline]
//!     fn from_seed(seed: Self::Seed) -> Self {
//!         // Always use little-endian byte order to ensure portable results
//!         let state = u32::from_le_bytes(seed);
//!         Self { state }
//!     }
//! }
//!
//! impl RngCore for Step32Rng {
//!     #[inline]
//!     fn next_u32(&mut self) -> u32 {
//!         // ...
//!         # let val = self.state;
//!         # self.state = val + 1;
//!         # val
//!     }
//!
//!     #[inline]
//!     fn next_u64(&mut self) -> u64 {
//!         utils::next_u64_via_u32(self)
//!     }
//!
//!     #[inline]
//!     fn fill_bytes(&mut self, dst: &mut [u8]) {
//!         utils::fill_bytes_via_next_word(dst, || self.next_u32());
//!     }
//! }
//!
//! # let mut rng = Step32Rng::seed_from_u64(42);
//! # assert_eq!(rng.next_u32(), 0x7ba1_8fa4);
//! # assert_eq!(rng.next_u64(), 0x7ba1_8fa6_7ba1_8fa5);
//! # let mut buf = [0u8; 5];
//! # rng.fill_bytes(&mut buf);
//! # assert_eq!(buf, [0xa7, 0x8f, 0xa1, 0x7b, 0xa8]);
//! ```
//!
//! ## RNG outputs `u64`
//!
//! ```
//! use rand_core::{RngCore, SeedableRng, utils};
//!
//! pub struct Step64Rng {
//!     state: u64
//! }
//!
//! impl SeedableRng for Step64Rng {
//!     type Seed = [u8; 8];
//!
//!     #[inline]
//!     fn from_seed(seed: Self::Seed) -> Self {
//!         // Always use little-endian byte order to ensure portable results
//!         let state = u64::from_le_bytes(seed);
//!         Self { state }
//!     }
//! }
//!
//! impl RngCore for Step64Rng {
//!     #[inline]
//!     fn next_u32(&mut self) -> u32 {
//!         self.next_u64() as u32
//!     }
//!
//!     #[inline]
//!     fn next_u64(&mut self) -> u64 {
//!         // ...
//!         # let val = self.state;
//!         # self.state = val + 1;
//!         # val
//!     }
//!
//!     #[inline]
//!     fn fill_bytes(&mut self, dst: &mut [u8]) {
//!         utils::fill_bytes_via_next_word(dst, || self.next_u64());
//!     }
//! }
//!
//! # let mut rng = Step64Rng::seed_from_u64(42);
//! # assert_eq!(rng.next_u32(), 0x7ba1_8fa4);
//! # assert_eq!(rng.next_u64(), 0x0a3d_3258_7ba1_8fa5);
//! # let mut buf = [0u8; 5];
//! # rng.fill_bytes(&mut buf);
//! # assert_eq!(buf, [0xa6, 0x8f, 0xa1, 0x7b, 0x58]);
//! ```
//!
//! ## RNG outputs `[u32; N]`
//!
//! ```
//! use rand_core::{RngCore, SeedableRng, utils};
//!
//! struct Block8x32RngInner {
//!     // ...
//!     # state: [u32; 8]
//! }
//!
//! impl Block8x32RngInner {
//!     fn new(seed: [u32; 8]) -> Self {
//!         // ...
//!         # Self { state: seed }
//!     }
//!
//!     fn next_block(&mut self, block: &mut [u32; 8]) {
//!         // ...
//!         # *block = self.state;
//!         # self.state.iter_mut().for_each(|v| *v += 1);
//!     }
//! }
//!
//! pub struct Block8x32Rng {
//!     inner: Block8x32RngInner,
//!     buffer: utils::BlockBuffer<u32, 8>,
//! }
//!
//! impl SeedableRng for Block8x32Rng {
//!     type Seed = [u8; 32];
//!
//!     #[inline]
//!     fn from_seed(seed: Self::Seed) -> Self {
//!         let seed: [u32; 8] = utils::read_words(&seed);
//!         Self {
//!             inner: Block8x32RngInner::new(seed),
//!             buffer: Default::default(),
//!         }
//!     }
//! }
//!
//! impl RngCore for Block8x32Rng {
//!     #[inline]
//!     fn next_u32(&mut self) -> u32 {
//!         self.buffer.next_word(|block| self.inner.next_block(block))
//!     }
//!
//!     #[inline]
//!     fn next_u64(&mut self) -> u64 {
//!         self.buffer.next_u64(|block| self.inner.next_block(block))
//!     }
//!
//!     #[inline]
//!     fn fill_bytes(&mut self, dst: &mut [u8]) {
//!         self.buffer.fill_bytes(dst, |block| self.inner.next_block(block));
//!     }
//! }
//!
//! # let mut rng = Block8x32Rng::seed_from_u64(42);
//! # assert_eq!(rng.next_u32(), 0x7ba1_8fa4);
//! # assert_eq!(rng.next_u64(), 0xcca1_b8ea_0a3d_3258);
//! # let mut buf = [0u8; 5];
//! # rng.fill_bytes(&mut buf);
//! # assert_eq!(buf, [0x69, 0x01, 0x14, 0xb8, 0x2b]);
//! ```
//!
//! ## RNG outputs `[u64; N]`
//!
//! ```
//! use rand_core::{RngCore, SeedableRng, utils};
//!
//! struct Block4x64RngInner {
//!     // ...
//!     # state: [u64; 4],
//! }
//!
//! impl Block4x64RngInner {
//!     fn new(seed: [u64; 4]) -> Self {
//!         // ...
//!         # Self { state: seed }
//!     }
//!
//!     fn next_block(&mut self, block: &mut [u64; 4]) {
//!         // ...
//!         # *block = self.state;
//!         # self.state.iter_mut().for_each(|v| *v += 1);
//!     }
//! }
//!
//! pub struct Block4x64Rng {
//!     inner: Block4x64RngInner,
//!     buffer: utils::BlockBuffer<u64, 4>,
//! }
//!
//! impl SeedableRng for Block4x64Rng {
//!     type Seed = [u8; 32];
//!
//!     #[inline]
//!     fn from_seed(seed: Self::Seed) -> Self {
//!         let seed: [u64; 4] = utils::read_words(&seed);
//!         Self {
//!             inner: Block4x64RngInner::new(seed),
//!             buffer: Default::default(),
//!         }
//!     }
//! }
//!
//! impl RngCore for Block4x64Rng {
//!     #[inline]
//!     fn next_u32(&mut self) -> u32 {
//!         self.next_u64() as u32
//!     }
//!
//!     #[inline]
//!     fn next_u64(&mut self) -> u64 {
//!         self.buffer.next_word(|block| self.inner.next_block(block))
//!     }
//!
//!     #[inline]
//!     fn fill_bytes(&mut self, dst: &mut [u8]) {
//!         self.buffer.fill_bytes(dst, |block| self.inner.next_block(block));
//!     }
//! }
//!
//! # let mut rng = Block4x64Rng::seed_from_u64(42);
//! # assert_eq!(rng.next_u32(), 0x7ba1_8fa4);
//! # assert_eq!(rng.next_u64(), 0xb814_0169_cca1_b8ea);
//! # let mut buf = [0u8; 5];
//! # rng.fill_bytes(&mut buf);
//! # assert_eq!(buf, [0x2b, 0x8c, 0xc8, 0x75, 0x18]);
//! ```
//!
//! ## RNG outputs bytes
//!
//! ```
//! use rand_core::RngCore;
//!
//! pub struct FillRng {
//!     // ...
//!     # state: u8,
//! }
//!
//! impl RngCore for FillRng {
//!     #[inline]
//!     fn next_u32(&mut self) -> u32 {
//!         let mut buf = [0; 4];
//!         self.fill_bytes(&mut buf);
//!         u32::from_le_bytes(buf)
//!     }
//!
//!     #[inline]
//!     fn next_u64(&mut self) -> u64 {
//!         let mut buf = [0; 8];
//!         self.fill_bytes(&mut buf);
//!         u64::from_le_bytes(buf)
//!     }
//!
//!     #[inline]
//!     fn fill_bytes(&mut self, dst: &mut [u8]) {
//!         // ...
//!         # for byte in dst {
//!         #     let val = self.state;
//!         #     self.state = val + 1;
//!         #     *byte = val;
//!         # }
//!     }
//! }
//!
//! # let mut rng = FillRng { state: 0 };
//! # assert_eq!(rng.next_u32(), 0x03_020100);
//! # assert_eq!(rng.next_u64(), 0x0b0a_0908_0706_0504);
//! # let mut buf = [0u8; 5];
//! # rng.fill_bytes(&mut buf);
//! # assert_eq!(buf, [0x0c, 0x0d, 0x0e, 0x0f, 0x10]);
//! ```
//!
//! Note that you can use `from_ne_bytes` instead of `from_le_bytes`
//! if your `fill_bytes` implementation is not reproducible.

pub use crate::block_buffer::BlockBuffer;
pub use crate::word::Word;

use crate::RngCore;

/// Implement `next_u64` via `next_u32` using little-endian order.
#[inline(always)]
pub fn next_u64_via_u32<R: RngCore + ?Sized>(rng: &mut R) -> u64 {
    // Use LE; we explicitly generate one value before the next.
    let x = u64::from(rng.next_u32());
    let y = u64::from(rng.next_u32());
    (y << 32) | x
}

/// Implement `fill_bytes` via `next_u64` using little-endian order.
#[inline]
pub fn fill_bytes_via_next_word<W: Word>(dst: &mut [u8], mut next_word: impl FnMut() -> W) {
    let mut chunks = dst.chunks_exact_mut(size_of::<W>());
    for chunk in &mut chunks {
        let val = next_word();
        chunk.copy_from_slice(val.to_le_bytes().as_ref());
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
        let val = next_word().to_le_bytes();
        rem.copy_from_slice(&val.as_ref()[..rem.len()]);
    }
}

/// Reads array of words from byte slice `src` using little endian order.
///
/// # Panics
/// If `size_of_val(src) != size_of::<[W; N]>()`.
#[inline(always)]
pub fn read_words<W: Word, const N: usize>(src: &[u8]) -> [W; N] {
    assert_eq!(size_of_val(src), size_of::<[W; N]>());
    let mut dst = [W::from_usize(0); N];
    let chunks = src.chunks_exact(size_of::<W>());
    for (out, chunk) in dst.iter_mut().zip(chunks) {
        let Ok(bytes) = chunk.try_into() else {
            unreachable!()
        };
        *out = W::from_le_bytes(bytes);
    }
    dst
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_read() {
        let bytes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

        let buf: [u32; 4] = read_words(&bytes);
        assert_eq!(buf[0], 0x04030201);
        assert_eq!(buf[3], 0x100F0E0D);

        let buf: [u32; 3] = read_words(&bytes[1..13]); // unaligned
        assert_eq!(buf[0], 0x05040302);
        assert_eq!(buf[2], 0x0D0C0B0A);

        let buf: [u64; 2] = read_words(&bytes);
        assert_eq!(buf[0], 0x0807060504030201);
        assert_eq!(buf[1], 0x100F0E0D0C0B0A09);

        let buf: [u64; 1] = read_words(&bytes[7..15]); // unaligned
        assert_eq!(buf[0], 0x0F0E0D0C0B0A0908);
    }
}
