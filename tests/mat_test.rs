use forge_cv::{Mat, ScalarType};

const TEST_IMAGE: &str = "/Users/zvistein/GitHub_repos/GeneratedOutput/BullsAI/data backup/2BulletData/images/train/fa1833bb-Screenshot_20220718-093648_Maps.jpg";

#[test]
fn imread_loads_image() {
    let m = Mat::imread(TEST_IMAGE);

    assert!(m.rows() > 0);
    assert!(m.cols() > 0);
    assert_eq!(m.channels(), 3);
    assert_eq!(m.scalar_type(), ScalarType::Uint8);
    assert_eq!(m.bytes(), m.rows() * m.cols() * 3);


    let mm = m.cvt_color_gray().convert_to(ScalarType::Float32, 1.0 / 255.0);

    assert_eq!(mm.rows(), m.rows());
    assert_eq!(mm.cols(), m.cols());
    assert_eq!(mm.channels(), 1);
    assert_eq!(mm.scalar_type(), ScalarType::Float32);
    assert_eq!(mm.bytes(), mm.rows() * mm.cols() * 4);

    println!("Loaded: {}x{} x{}ch  ({} bytes)",
        m.rows(), m.cols(), m.channels(), m.bytes());
}

#[test]
fn imread_pixel_access() {
    let m = Mat::imread(TEST_IMAGE);

    // Read first pixel
    let r = m.value_at(0, 0, 0);
    let g = m.value_at(0, 0, 1);
    let b = m.value_at(0, 0, 2);

    assert!((0.0..=255.0).contains(&r));
    assert!((0.0..=255.0).contains(&g));
    assert!((0.0..=255.0).contains(&b));

    println!("Pixel [0,0] = ({}, {}, {})", r, g, b);
}

#[test]
fn roundtrip_from_slice() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let m = Mat::from_slice("test", 2, 3, ScalarType::Float32, 1, &data);

    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 3);
    assert_eq!(m.as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn debug_fields_in_sync() {
    let data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
    let m = Mat::from_slice("sync_test", 2, 2, ScalarType::Float32, 1, &data);

    assert_eq!(m.data_len, m.bytes());
    assert_eq!(m.data_ptr, m.data.as_ptr() as usize);
}

#[test]
fn resize_updates_debug_fields() {
    let mut m = Mat::new("empty");
    assert_eq!(m.data_len, 0);

    m.resize(4, 4, ScalarType::Float32, 1);
    assert_eq!(m.data_len, 4 * 4 * 4); // 4x4 float32
    assert_eq!(m.data_ptr, m.data.as_ptr() as usize);
}
