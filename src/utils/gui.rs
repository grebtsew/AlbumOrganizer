use image::{DynamicImage, GenericImageView};
use minifb::{Key, Window, WindowOptions};

pub fn display_images(image_paths: Vec<String>, show: Option<bool>) {

    let def_show = show.unwrap_or(true);

    for path in &image_paths {
        let image = image::open(path).unwrap();

        if def_show {
            show_image(image);
        }
    }
}


pub fn show_image(img: DynamicImage) -> Result<(), String> {
    // Get the width and height of the image
    let (width, height) = img.dimensions();

    // Convert the image to a vector of u32 pixels
    let pixel_data = img.into_rgba8().into_raw();
    let pixel_data_u32: Vec<u32> = pixel_data
        .chunks_exact(4)
        .map(|chunk| {
            let r = chunk[0] as u32;
            let g = chunk[1] as u32;
            let b = chunk[2] as u32;
            let a = chunk[3] as u32;
            (a << 24) | (r << 16) | (g << 8) | b
        })
        .collect();

    // Create a window to display the image
    let mut window = Window::new(
        "Image Viewer",
        width as usize,
        height as usize,
        WindowOptions::default(),
    )
    .map_err(|e| format!("{}", e))?;

    // Loop until the window is closed
    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Update the window with the pixel data
        window
            .update_with_buffer(&pixel_data_u32, width as usize, height as usize)
            .map_err(|e| format!("{}", e))?;
    }

    Ok(())
}