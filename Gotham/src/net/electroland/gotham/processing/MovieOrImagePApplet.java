package net.electroland.gotham.processing;

import java.io.File;

import processing.core.PImage;


public class MovieOrImagePApplet extends GothamPApplet {

    private static final long serialVersionUID = 1L;
    private PImage image;
//    private Movie movie;

    @Override
    public void drawELUContent() {
        // based on the file name either draw an Image or render a movie.
        if (image != null){
            this.image(image, 0, 0, this.getSyncArea().width, this.getSyncArea().height);
        }// else if (movie != null) {
            // render movie here
        //}
    }

    @Override
    public void fileReceived(File file){
        String name = file.getAbsolutePath();
        System.out.println("loading " + name);
        if (name.toLowerCase().endsWith(".mov")){
            image = null;
            // load movie here
        } else if (name.toLowerCase().endsWith(".jpg") || name.toLowerCase().endsWith(".png")) {
            // movie = null;
            image = loadImage(file.getAbsolutePath());
        }
    }
}
