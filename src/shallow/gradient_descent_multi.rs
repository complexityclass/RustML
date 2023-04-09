use ndarray::{s, Array2, ArrayView2};
use std::f64::consts;

pub fn gradient_descent(
    x: &Array2<f64>,
    y: &Array2<f64>,
    w_in: &Array2<f64>,
    b_in: f64,
    alpha: f64,
    num_iters: usize,
) -> (Array2<f64>, f64) {
    let mut b = b_in;
    let mut w = w_in.clone();

    for i in 0..num_iters {
        let (dj_dw, dj_db) = compute_gradient(x, y, &w, b);
        w = &w - alpha * &dj_dw;
        b = b - alpha * dj_db;

        if i % 100 == 0 {
            let cost = compute_cost(&x, &y, &w, b);
            println!("Iteration {:4}: Cost {:0.2e} dj_dw: {:0.3e}, dj_db: {:0.3e}  w: {:0.3e}, b: {:0.5e}",
                     i, cost, dj_dw, dj_db, w, b);
        }
    }
    (w, b)
}

fn compute_cost(
    x: &Array2<f64>,
    y: &Array2<f64>,
    w: &Array2<f64>,
    b: f64
) -> f64 {
    assert_eq!(x.shape()[0], y.shape()[0]);
    let m = x.shape()[0];
    let mut cost = 0f64;
    for i in 0..m {
        let d = w.dot(&x.slice(s![i, ..]));
        let f_wb: f64 = d[0] + b;
        cost = cost + (f_wb - y[[i, 0]]).powf(2.0);
    }
    cost / (2.0 * m as f64)
}

fn compute_gradient(
    x: &Array2<f64>,
    y: &Array2<f64>,
    w: &Array2<f64>,
    b: f64
) -> (ndarray::Array2<f64>, f64) {
    assert_eq!(x.shape()[0], y.shape()[0]);
    let mut dj_dw = Array2::<f64>::zeros((1, x.shape()[1]));
    let mut dj_db = 0f64;
    let m = x.shape()[0];

    for i in 0..m {
        let err = w.dot(&x.slice(s![i, ..]))[0] + b - y[[i, 0]];
        for j in 0..x.shape()[1] {
            dj_dw[[0, j]] += err * x[[i, j]];
        }
        dj_db += err;
    }
    dj_dw /= m as f64;
    dj_db /= m as f64;

    (dj_dw, dj_db)
}

mod tests {
    use ndarray::array;
    use super::*;

    #[test]
    fn test_compute_cost() {
        let x = array![
            [2104.0, 5.0, 1.0, 45.0],
            [1416.0, 3.0, 2.0, 40.0],
            [852.0, 2.0, 1.0, 35.0]
        ];
        let row_y = array![460.0, 232.0, 178.0];
        let y: Array2<f64> = row_y.into_shape((3, 1)).unwrap();
        let b = 785.1811367994083;
        let row_w = array![0.39133535, 18.75376741, -53.36032453, -26.42131618];
        let w: Array2<f64> = row_w.into_shape((1, 4)).unwrap();
        let cost = compute_cost(&x, &y, &w, b);
        let expected = 1.5578904880036537e-12;
        print!("cost: {}", cost);
        assert_eq!((cost - expected).abs() < 1e-4, true);
    }

    #[test]
    fn test_compute_gradient() {
        let x = array![
            [2104.0, 5.0, 1.0, 45.0],
            [1416.0, 3.0, 2.0, 40.0],
            [852.0, 2.0, 1.0, 35.0]
        ];
        let row_y = array![460.0, 232.0, 178.0];
        let y: Array2<f64> = row_y.into_shape((3, 1)).unwrap();
        let b = 785.1811367994083;
        let row_w = array![0.39133535, 18.75376741, -53.36032453, -26.42131618];
        let w: Array2<f64> = row_w.into_shape((1, 4)).unwrap();
        let (dj_dw, dj_db) = compute_gradient(&x, &y, &w, b);
        let expected_dj_dw = array![[-2.73e-03, -6.27e-06, -2.22e-06, -6.92e-05]];
        let expected_dj_db = -1.673925169143331e-06;
        println!("{:?}", dj_dw.shape());
        for i in 0..dj_dw.shape()[1] {
            assert_eq!((dj_dw[[0, i]] - expected_dj_dw[[0, i]] as f64).abs() < 1e-4, true);
        }
        assert_eq!((dj_db - expected_dj_db).abs() < 1e-4, true);
    }

    #[test]
    fn test_gradient_descent() {
        let x = array![
            [2104.0, 5.0, 1.0, 45.0],
            [1416.0, 3.0, 2.0, 40.0],
            [852.0, 2.0, 1.0, 35.0]
        ];
        let row_y = array![460.0, 232.0, 178.0];
        let y: Array2<f64> = row_y.into_shape((3, 1)).unwrap();
        let b = 0.0;
        let row_w = array![0.0, 0.0, 0.0, 0.0];
        let w: Array2<f64> = row_w.into_shape((1, 4)).unwrap();
        let alpha = 5.0e-7;
        let num_iters = 1000;
        let (w, b) = gradient_descent(&x, &y, &w, b, alpha, num_iters);
        let expected_w = array![[0.2, 0.0, -0.01, -0.07]];
        let expected_b = 0.0;
        for i in 0..w.shape()[1] {
            assert_eq!((w[[0, i]] - expected_w[[0, i]] as f64).abs() < 1e-4, true);
        }
        assert_eq!((b - expected_b).abs() < 1e-4, true);
    }
}
