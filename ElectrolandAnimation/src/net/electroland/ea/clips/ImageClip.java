package net.electroland.ea.clips;

import java.awt.Graphics;
import java.awt.Image;
import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.Map;
import java.util.TreeSet;
import java.util.Vector;

import javax.imageio.ImageIO;

import net.electroland.ea.Clip;
import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;

public class ImageClip extends Clip {

    @SuppressWarnings("unused")
    private Map<String, Object> context;
    private Image[] frames;
    private int pointer = 0;

    @Override
    public void config(ParameterMap primaryParams,
            Map<String, ParameterMap> extendedParams) {

        String base = primaryParams.getRequired("root");
        Vector<Image> framesTmp = new Vector<Image>();

        // load all specified images
        TreeSet<String> alphabatizedNames = new TreeSet<String>();
        alphabatizedNames.addAll(extendedParams.keySet());
        for (String name : alphabatizedNames)
        {
            if (name.startsWith("frame."))
            {
                try {
                    framesTmp.add(ImageIO.read(new File(base, name)));
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
        this.context = context;
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
            g.drawImage(frames[pointer++],
                        0,
                        0,
                        this.getBaseDimensions().width,
                        this.getBaseDimensions().height,
                        null);

            if (pointer == frames.length)
                pointer = 0;
        }

        return image;
    }
}