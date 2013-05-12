package net.electroland.ea;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.image.BufferedImage;
import java.util.Hashtable;
import java.util.Map;
import java.util.Queue;
import java.util.Vector;
import java.util.concurrent.LinkedBlockingQueue;

import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;

import org.apache.log4j.Logger;

/**
 * Animation generates frames of animation any time you call getFrame(). Within
 * that frame are any number of "Clips", which are basically content with that
 * can be moved around the stage using tweening directives.
 * 
 * This class does not run any kind of animation thread.  It's up to you to
 * poll it per frame you want to render.  It animates all clips on the stage
 * based on absolute time, so the rate at which you call on frames does not
 * affect how fast clips make it to their directed final states.
 * 
 * See net.electroland.ea.test.AnimationTestFrame for an example of how to use 
 * this class properly.
 * 
 * @author production
 *
 */
public class Animation {

    private static Logger logger = Logger.getLogger(Animation.class);

    private Clip rootClip;
    private Dimension frameDimemsions;
    private Map<String, Content>contentPrototypes;
    private Vector<AnimationListener>listeners;
    private Queue<Object>messageQueue;

    public Animation()
    {
        listeners = new Vector<AnimationListener>();
    }

    public Animation(String propsName)
    {
        listeners = new Vector<AnimationListener>();
        config(propsName);
    }

    public void load(ElectrolandProperties p){

        logger.info("loading...");
        frameDimemsions = new Dimension(p.getRequiredInt("settings", "global", "width"),
                                        p.getRequiredInt("settings", "global", "height"));
        rootClip = new Clip(null, Color.GRAY, 0, 0, frameDimemsions.width, frameDimemsions.height, 1.0f);
        rootClip.animationManager = this;

        messageQueue = new LinkedBlockingQueue<Object>();

        // clip
        contentPrototypes = new Hashtable<String,Content>();
        Map<String, ParameterMap> contentParams = p.getObjects("content");

        for (String s : contentParams.keySet()){
            logger.info("loading content '" + s + "'");
            ParameterMap universalParams = contentParams.get(s);
            
            Map<String, ParameterMap> extendedParams = null;
            try{
                extendedParams = p.getObjects(s);
            }catch(OptionException e){
                // kludgy.  Most likely thrown if there aren't any extended params,
                // but should more gracefully check for that.
            }

            Content content = (Content)(universalParams.getRequiredClass("class"));
            content.config(universalParams, extendedParams);
            contentPrototypes.put(s, content);
        }
    }

    public void config(String propsName)
    {
        logger.info("loading " + propsName);
        ElectrolandProperties p = new ElectrolandProperties(propsName);
        load(p);
    }

    public void setBackground(Color bg)
    {
        this.rootClip.currentState.bgcolor = bg;
    }
    public Dimension getFrameDimensions() {
        return frameDimemsions;
    }
    public void setFrameDimensions(Dimension dimensions) {
        this.frameDimemsions = dimensions;
    }
    public void addListener(AnimationListener a){
        this.listeners.add(a);
    }
    public void removeListener(AnimationListener a){
        this.listeners.remove(a);
    }

    public Clip addClip(int x, int y, int width, int height, float alpha)
    {
        return rootClip.addClip(x, y, width, height, alpha);
    }
    public Clip addClip(Content c, int x, int y, int width, int height, float alpha)
    {
        return rootClip.addClip(c, x, y, width, height, alpha);
    }
    public Clip addClip(Content c, Color bgcolor, int x, int y, int width, int height, float alpha)
    {
        return rootClip.addClip(c, bgcolor, x , y, width, height, alpha);
    }
    public Clip addClip(Color bgcolor, int x, int y, int width, int height, float alpha)
    {
        return rootClip.addClip(bgcolor, x, y, width, height, alpha);
    }
    public Content getContent(String contentId)
    {
        Object proto = contentPrototypes.get(contentId);
        if (proto == null){
            throw new RuntimeException("no content named '" +  contentId + "'.");
        }else{
            return (Content)((Content)proto).clone();
        }
    }
    public BufferedImage getFrame()
    {
        BufferedImage bi = rootClip.getImage();
        processMessageQueue();
        return bi;
    }
    public static int[] toPixels(BufferedImage stage, int width, int height)
    {
        int[] pixels = new int[width * height];
        ((BufferedImage)stage).getRGB(0, 0, width, height, pixels, 0, width);
        return pixels;
    }

    protected void announce(Object message){
        messageQueue.add(message);
    }

    protected void processMessageQueue(){
        if (messageQueue.size() > 0){
            Object message = messageQueue.poll();
            for (AnimationListener a : listeners){
                a.messageReceived(message);
            }
        }
    }
}