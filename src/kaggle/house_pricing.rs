use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use shallow::gradient_descent_single_param::gradient_descent;
use crate::shallow;

pub fn predict_price() -> Result<(), Box<dyn Error>> {
    let (w, b) = train()?;
    println!("w: {:?}, b: {:?}", w, b);
    predict(w, b);
    Ok(())
}

fn train() -> Result<(f64, f64), Box<dyn Error>> {
    let path = Path::new("data/train.csv");
    let file = File::open(&path)?;
    let mut reader = ReaderBuilder::new()
        .delimiter(b',')
        .quote(b'"')
        .double_quote(true)
        .flexible(true)
        .from_reader(file);

    let mut x_train: Vec<f64> = Vec::new();
    let mut y_train: Vec<f64> = Vec::new();

    for result in reader.records() {
        let record = result?;
        // println!("{:?}", record);
        let num_params = record.len();
        let area_str: String = record[3].to_string();

        if area_str == "NA" {
            continue;
        }

        let area: f64 = area_str.parse()?;
        let price: f64 = record[num_params - 1].parse()?;

        x_train.push(area);
        y_train.push(price);
    }

    let (w, b) = gradient_descent(&x_train, &y_train, 0.0, 0.0, 0.0001, 10000);
    Ok((w, b))
}

fn predict(w: f64, b: f64) -> Result<(), Box<dyn Error>> {
    let path = Path::new("data/test.csv");
    let file = File::open(&path)?;
    let mut reader = ReaderBuilder::new()
        .delimiter(b',')
        .quote(b'"')
        .double_quote(true)
        .flexible(true)
        .from_reader(file);

    let write_path = Path::new("data/predict.csv");
    let mut wtr = csv::WriterBuilder::new().has_headers(true).from_path(write_path)?;
    let headers = vec!["Id", "SalePrice"];
    wtr.write_record(&headers)?;
    for result in reader.records() {
        let record = result?;
        let id = record[0].parse::<i32>().unwrap();
        let area_str: String = record[3].to_string();
        let area = if area_str == "NA" {
            70.04
        } else {
            area_str.parse().unwrap()
        };

        let price = w * area + b;
        println!("id: {:?} area {:?}, price: {:?} ", id, area, price);
        wtr.write_record(&[id.to_string(), price.to_string()])?;
    }
    Ok(())
}