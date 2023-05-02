#[path = "../src/utils/file.rs"]
pub mod file;

#[path = "../src/utils/gui.rs"]
pub mod gui;


#[test]
fn test_find_all_images_from_folder_and_show() {
    let album_path = "./data/test_images";

    let image_paths = file::find_images(album_path);

    println!("{:?}", image_paths);

    println!("Amount of images found {}", image_paths.len());
    assert!(image_paths.len() == 5);

    gui::display_images(image_paths, Some(false));
    assert!(true)
}




    
