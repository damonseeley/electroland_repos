package net.electroland.gotham.processing;

import java.io.File;

import processing.core.PImage;
import processing.video.*;

public class MovieOrImagePApplet extends GothamPApplet {

    private static final long serialVersionUID = 1L;
    private PImage image;
    private Movie movie;

    @Override
    public void drawELUContent() {
        // based on the file name either draw an Image or render a movie.
        if (image != null){
            this.image(image, 0, 0, this.getSyncArea().width, this.getSyncArea().height);
        } else if (movie != null) {
            this.image(movie, 0, 0, this.getSyncArea().width, this.getSyncArea().height);
        }
    }

    @Override
    public void fileReceived(File file){
        String name = file.getAbsolutePath();
        if (name.toLowerCase().endsWith(".mov") || name.toLowerCase().endsWith(".mp4")){
            if (movie != null){
                movie.stop();
            }
            movie = new Movie(this, name);
            movie.loop();
            image = null;
            System.out.println("loaded movie " + movie);
        } else if (name.toLowerCase().endsWith(".jpg") || name.toLowerCase().endsWith(".png")) {
            image = loadImage(file.getAbsolutePath());
            if (movie != null){
                movie.stop();
            }
            movie = null;
        }
    }

    public void movieEvent(Movie m) {
        m.read();
      }

}
