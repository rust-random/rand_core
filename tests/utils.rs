use rand_core::utils::read_words;

#[test]
fn test_read_words() {
    let bytes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

    let buf: [u32; 4] = read_words(&bytes);
    assert_eq!(buf[0], 0x0403_0201);
    assert_eq!(buf[3], 0x100F_0E0D);

    let buf: [u32; 3] = read_words(&bytes[1..13]); // unaligned
    assert_eq!(buf[0], 0x0504_0302);
    assert_eq!(buf[2], 0x0D0C_0B0A);

    let buf: [u64; 2] = read_words(&bytes);
    assert_eq!(buf[0], 0x0807_0605_0403_0201);
    assert_eq!(buf[1], 0x100F_0E0D_0C0B_0A09);

    let buf: [u64; 1] = read_words(&bytes[7..15]); // unaligned
    assert_eq!(buf[0], 0x0F0E_0D0C_0B0A_0908);
}
