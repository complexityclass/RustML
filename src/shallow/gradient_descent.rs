use std::f64::consts;
use std::io::repeat;

pub fn gradient_descent(
    x: &[f64],
    y: &[f64],
    w_in: f64,
    b_in: f64,
    alpha: f64,
    num_iters: usize,
) -> (f64, f64) {
    let mut b = b_in;
    let mut w = w_in;

    for i in 0..num_iters {
        let (dj_dw, dj_db) = compute_gradient(x, y, w, b);
        w = w - alpha * dj_dw;
        b = b - alpha * dj_db;

        if i % 100 == 0 {
            let cost = compute_cost(&x, &y, w, b);
            println!("Iteration {:4}: Cost {:0.2e} dj_dw: {:0.3e}, dj_db: {:0.3e}  w: {:0.3e}, b: {:0.5e}",
                     i, cost, dj_dw, dj_db, w, b);
        }
    }
    (w, b)
}

fn compute_cost(x: &[f64], y: &[f64], w: f64, b: f64) -> f64 {
    assert_eq!(x.len(), y.len());
    let m = x.len();
    let mut cost = 0f64;
    for i in 0..m {
        let f_wb = w * x[i] + b;
        cost = cost + (f_wb - y[i]).powf(2.0);
    }

    1.0 / (2.0 * m as f64) * cost
}

fn compute_gradient(x: &[f64], y: &[f64], w: f64, b: f64) -> (f64, f64) {
    assert_eq!(x.len(), y.len());
    let mut dj_dw = 0f64;
    let mut dj_db = 0f64;
    let m = x.len();

    for i in 0..m {
        let f_wb = w * x[i] + b;
        let dj_dw_i = (f_wb - y[i]) * x[i];
        let dj_db_i = f_wb - y[i];
        dj_db += dj_db_i;
        dj_dw += dj_dw_i;
    }

    dj_dw /= m as f64;
    dj_db /= m as f64;

    (dj_dw, dj_db)
}

mod tests {
    use super::*;

    #[test]
    fn test_compute_cost() {
        let x = vec![1.0, 2.0];
        let y = vec![300.0, 500.0];
        let w = 1.0;
        let b = 1.0;
        let cost = compute_cost(&x, &y, w, b);
        assert_eq!(cost, 83953.25);
    }

    #[test]
    fn test_compute_gradient() {
        let x = vec![1.0, 2.0];
        let y = vec![300.0, 500.0];
        let w = 2.0;
        let b = 2.0;
        let (dj_dw, dj_db) = compute_gradient(&x, &y, w, b);
        assert_eq!(dj_dw, -648.0);
        assert_eq!(dj_db, -399.0);
    }

    #[test]
    fn test_gradient_descent() {
        let x = vec![1.0, 2.0];
        let y = vec![300.0, 500.0];
        let w = 0.0;
        let b = 0.0;
        let (w_f, b_f) = gradient_descent(&x, &y, w, b, 0.01, 10000);
        println!("(w,b) found by gradient descent: ({:8.4},{:8.4})", w_f, b_f);
    }
}
