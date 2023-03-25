mod shallow;
mod kaggle;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    kaggle::house_pricing::predict_price()
}
