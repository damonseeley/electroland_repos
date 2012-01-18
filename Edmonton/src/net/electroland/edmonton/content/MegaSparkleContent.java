package net.electroland.edmonton.content;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.util.Map;
import java.util.Vector;

import net.electroland.ea.Content;
import net.electroland.utils.ParameterMap;

import org.apache.log4j.Logger;

public class MegaSparkleContent extends Content {

    private static Logger logger = Logger.getLogger(MegaSparkleContent.class);
    int sparkleWidth;
    double lastChange;
    int colorArraySize;
    Vector<Integer> colors;
    double sparkleRate;
    int sparkles;
    private Dimension baseDimensions;

    @Override
    public void renderContent(BufferedImage image) {
        if ((System.currentTimeMillis() - lastChange ) > sparkleRate) {
            incrementSparkles();
            lastChange = System.currentTimeMillis();
        }
        
        double pctChanged = (System.currentTimeMillis() - lastChange )/sparkleRate;

        // this looks wrong - Bradley
        // I'd expect:
        BufferedImage bi = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_INT_RGB);
       // BufferedImage bi = new BufferedImage(baseDimensions.width, image.getHeight(null), BufferedImage.TYPE_INT_RGB);
        Graphics g2 = bi.getGraphics();

        for (int i=1; i < sparkles; i++){
            int newx = i*sparkleWidth;
            int clr = 0;
            
            if (colors.get(i) > colors.get(i-1)) {
                clr = colors.get(i-1) + (int)((Math.abs(colors.get(i) - colors.get(i-1)))*pctChanged);
            } else {
                clr = colors.get(i-1) - (int)((Math.abs(colors.get(i) - colors.get(i-1)))*pctChanged);
            }
            
            //clr = colors.get(i);
            g2.setColor(new Color(clr,clr,clr));
            g2.fillRect(newx, 0, sparkleWidth, 16);
        }

        Graphics g = image.getGraphics();
        g.drawImage(bi, 0, 0, null);
    }

    @Override
    public void config(ParameterMap primaryParams,
            Map<String, ParameterMap> extendedParams) {
        sparkleWidth = primaryParams.getRequiredInt("sparkleWidth");
        sparkleRate = primaryParams.getRequiredInt("sparkleTime"); //ms delay

        baseDimensions = new Dimension(primaryParams.getRequiredInt("width"), primaryParams.getRequiredInt("height"));
        
        lastChange = System.currentTimeMillis();
        //sparkleWidth = this.getBaseDimensions().width; //returns 320 because that's what it's created as
        sparkles = baseDimensions.width/sparkleWidth;

        //fill colors
        colorArraySize = sparkles*2; //just to be safe
        colors = new Vector<Integer>(colorArraySize);
        makeSparkles();
        incrementSparkles();
    }

    @Override
    public void init(Map<String, Object> context) {
        // do nothing
        // send a soundcontroller command probably here
    }

    private void makeSparkles() {
        colors = new Vector<Integer>(colorArraySize);
        for (int i=0; i<colorArraySize; i++){
            int newClr = (int)(Math.random()*128) + 128;
            colors.add(newClr);
        }
    }

    private void incrementSparkles() {
        // clone the array and move everything forward

        Vector<Integer> newColors = new Vector<Integer>(colorArraySize);
        for (int i=1; i<colors.size(); i++){
            newColors.add(colors.get(i));
        }
        newColors.add((int)(Math.random()*128) + 128);
        colors = newColors;
    }
}
