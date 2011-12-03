package net.electroland.ea.clips;

import java.awt.Graphics;
import java.awt.Image;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.TreeSet;
import java.util.Vector;

import javax.imageio.ImageIO;

import net.electroland.ea.Clip;
import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;

import org.apache.log4j.Logger;

public class ImageClip extends Clip {

    private static Logger logger = Logger.getLogger(ImageClip.class);
    private Image[] frames;
    private int pointer = 0;
    int delay;
    long lastRender;

    @Override
    public void config(ParameterMap primaryParams,
            Map<String, ParameterMap> extendedParams) {

        delay = 1000 / primaryParams.getRequiredInt("fps");

        String base = primaryParams.getRequired("root");
        Vector<Image> framesTmp = new Vector<Image>();

        // load all specified images
        // images will be played in order of their frame name, using natural
        // sort.  E.g., frame1, frame2 ... frame9, frame 10
        TreeSet<String> alphabatizedNames = new TreeSet<String>(new AlphanumComparator());
        alphabatizedNames.addAll(extendedParams.keySet());
        for (String name : alphabatizedNames)
        {
            if (name.startsWith("frame."))
            {
                try {
                    String filename = extendedParams.get(name).getRequired("file");
                    logger.info("loading image " + filename + " from " + base);
                    framesTmp.add(ImageIO.read(new File(base, filename)));
                } catch (IOException e) {
                    throw new OptionException(e);
                }
            }
        }
        frames = new Image[framesTmp.size()];
        framesTmp.toArray(frames);
    }

    @Override
    public void init(Map<String, Object> context) {
        // do nothing
    }

    @Override
    public boolean isDone() {
        // play forever, or until someone kills this clip manually
        return false;
    }

    @Override
    public Image getFrame(Image image) {

        if (frames.length > 0)
        {
            Graphics g = image.getGraphics();
            g.drawImage(frames[pointer],
                        0,
                        0,
                        this.getBaseDimensions().width,
                        this.getBaseDimensions().height,
                        null);
            g.dispose();

            long currentRender = System.currentTimeMillis();
            if (currentRender - lastRender > delay)
            {
                lastRender = currentRender;
                pointer++;
                if (pointer == frames.length)
                    pointer = 0;
            }
        }

        return image;
    }
}