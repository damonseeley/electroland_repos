package net.electroland.ea.content;

import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Map;

import javax.imageio.ImageIO;

import net.electroland.ea.Content;
import net.electroland.utils.ParameterMap;

import org.apache.log4j.Logger;

public class ImageContent extends Content {

    private static Logger logger = Logger.getLogger(ImageContent.class);
    private Image image;

    /**
     * a constructor for creating an ImageContent object from an
     * Image object rather than loading the image from animation.properties.
     * @param image
     */
    public void fromImage(Image image) {
        this.image = image;
    }

    @Override
    public void renderContent(BufferedImage canvas) {
        Graphics g = canvas.getGraphics();
        g.drawImage(image,
                    0,
                    0,
                    canvas.getWidth(),
                    canvas.getHeight(),
                    null);
        g.dispose();
    }

    @Override
    public void config(ParameterMap primaryParams,
            Map<String, ParameterMap> extendedParams) {
        String filename = primaryParams.getRequired("file");
        try {
            logger.trace("loading image " + filename);
            image = ImageIO.read(this.getClass().getClassLoader().getResourceAsStream(filename));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void init(Map<String, Object> context) {
        // do nothing
    }
}