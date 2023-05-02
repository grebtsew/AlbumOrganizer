mod utils { 
    pub mod file;
    pub mod gui;
}

use polars::prelude::*;


fn main() {

    // Load images
    let album_path = "./data/test_images";

    let image_paths = utils::file::find_images(album_path);
    println!("{:?}", image_paths);
    println!("Amount of images found {}", image_paths.len());


    // Store in database
    let mut persons: Vec<&str> = Vec::new();
    let mut boxes: Vec<Series> = Vec::new();

    for path in &image_paths {
        persons.push("");
        boxes.push(Series::new("d",&[0,0,0,0]));
    }

    let s0 = Series::new("image_paths", &image_paths);
    let s1 = Series::new("persons", &persons);
    let s2 = Series::new("boxes", &boxes);

    let df = DataFrame::new(vec![s0, s1, s2]);

    println!("{:?}", df.unwrap());

    // Show images (test open all images)
    utils::gui::display_images(image_paths, Some(false));

    // Start gui
    utils::gui::start_gui()
}
