package net.electroland.ea;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.image.BufferedImage;
import java.util.Collection;
import java.util.Hashtable;
import java.util.Map;

import net.electroland.ea.content.SolidColorContent;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;

import org.apache.log4j.Logger;

public class AnimationManager {

    private static Logger logger = Logger.getLogger(AnimationManager.class);
    private int fps;
    private Clip stage;
    private Dimension stageDimensions;
    private Map<String, Content>protoContent;
    private Map<String, Collection<Clip>>taggedClips;

    public int getFps() {
        return fps;
    }
    public void setFps(int fps) {
        this.fps = fps;
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
        stage = new Clip(new SolidColorContent(Color.white), 0, 0, stageDimensions.width, stageDimensions.height, 1.0);
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
    public Clip addClip(Clip c, int x, int y, int width, int height, double alpha)
    {
        return stage.addClip(c, x,y,width,height,alpha);
    }
    public Content getContent(String contentId)
    {
        return (Content)protoContent.get(contentId).clone();
    }
    public Collection<Clip> getClip(String tag)
    {
        return taggedClips.get(tag);
    }
    public BufferedImage getStage()
    {
        stage.processChanges();
        return stage.getImage(new BufferedImage(stageDimensions.width,
                                                stageDimensions.height,
                                                BufferedImage.TYPE_INT_ARGB));
    }
}