package net.electroland.ea;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.image.BufferedImage;
import java.util.Hashtable;
import java.util.Map;

import net.electroland.ea.content.SolidColorContent;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;

import org.apache.log4j.Logger;

/**
 * This controls all animation. It does not run any kind of animation thread,
 * even though it presents an FPS variable.  FPS is purely for passing on what
 * the user specified in the configuration file as the desired FPS. As a rule,
 * users should instatiate this object with the pointer to the property file
 * that defines Content objects and then:
 * 
 * getContet("myContent") to retrieve Content defined in the props file.
 * addClip(...) to put a new Clip on the stage
 * getStage() to get the Image that represents the stage at any point in time.
 * getStage() polls the system for the next state.
 * toPixels(...) to convert the stage to a pixel grab for use with ELU
 * 
 * @author production
 *
 */
public class AnimationManager {

    private static Logger logger = Logger.getLogger(AnimationManager.class);
    private int fps;
    private Clip stage;
    private Dimension stageDimensions;
    private Map<String, Content>protoContent;

    public int getFps() {
        return fps;
    }
    public void setFps(int fps) {
        this.fps = fps;
    }
    public void setStageColor(Color color)
    {
        this.stage.content = new SolidColorContent(color);
    }
    public Dimension getStageDimensions() {
        return stageDimensions;
    }
    public void setStageDimensions(Dimension stageDimensions) {
        this.stageDimensions = stageDimensions;
    }
    public AnimationManager(String propsName)
    {
        config(propsName);
    }
    public void config(String propsName)
    {
        logger.info("loading " + propsName);
        ElectrolandProperties p = new ElectrolandProperties(propsName);

        this.fps = p.getRequiredInt("settings", "global", "fps");

        stageDimensions = new Dimension(p.getRequiredInt("settings", "global", "width"),
                                p.getRequiredInt("settings", "global", "height"));
        stage = new Clip(new SolidColorContent(null), 0, 0, stageDimensions.width, stageDimensions.height, 1.0);
        // clip
        protoContent = new Hashtable<String,Content>();
        Map<String, ParameterMap> contentParams = p.getObjects("content");
        for (String s : contentParams.keySet()){
            logger.info("loading content '" + s + "'");
            ParameterMap universalParams = contentParams.get(s);

            Map<String, ParameterMap> extendedParams = null;
            try{
                extendedParams = p.getObjects(s);
            }catch(OptionException e){
                // not a problem.  There might not be any extended params.
            }

            Content content = (Content)(universalParams.getRequiredClass("class"));
            content.config(universalParams, extendedParams);
            protoContent.put(s, content);
        }
    }
    public Clip addClip(Content c, int x, int y, int width, int height, double alpha)
    {
        return stage.addClip(c, x,y,width,height,alpha);
    }
    public Clip addClip(Content c, int x, int y, int width, int height, double alpha, int delay)
    {
        return stage.addClip(c, x,y,width,height,alpha, delay);
    }
    public Clip addClip(Clip c, int x, int y, int width, int height, double alpha)
    {
        return stage.addClip(c, x,y,width,height,alpha);
    }
    public Content getContent(String contentId)
    {
        return (Content)protoContent.get(contentId).clone();
    }
    public BufferedImage getStage()
    {
        stage.processChanges();
        return stage.getImage(new BufferedImage(stageDimensions.width,
                                                stageDimensions.height,
                                                BufferedImage.TYPE_INT_ARGB), 1.0, 1.0);
    }
    public static int[] toPixels(BufferedImage stage, int width, int height)
    {
        int[] pixels = new int[width * height];
        ((BufferedImage)stage).getRGB(0, 0, width, height, pixels, 0, width);
        return pixels;
    }
}