mod utils { 
    pub mod file;
    pub mod gui;
}


fn main() {

    let album_path = "./data/test_images";

    let image_paths = utils::file::find_images(album_path);
    println!("{:?}", image_paths);
    println!("Amount of images found {}", image_paths.len());

    utils::gui::display_images(image_paths, None);
}
