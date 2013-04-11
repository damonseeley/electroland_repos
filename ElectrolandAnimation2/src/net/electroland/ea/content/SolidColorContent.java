package net.electroland.ea.content;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.util.Map;

import net.electroland.ea.Content;
import net.electroland.utils.ParameterMap;

public class SolidColorContent extends Content {

    private Color color;

    public Color getColor() {
        return color;
    }
    public void setColor(Color color) {
        this.color = color;
    }
    public SolidColorContent(){}

    public SolidColorContent(Color c){
        this.color = c;
    }
    @Override
    public void renderContent(BufferedImage canvas) {
        if (color != null){
            Graphics g = canvas.getGraphics();
            g.setColor(color);
            g.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
            g.dispose();
        }
    }
    @Override
    public void config(ParameterMap primaryParams,
            Map<String, ParameterMap> extendedParams) {
        if (primaryParams.getOptionalInt("red") != null &&
            primaryParams.getOptionalInt("green") != null && 
            primaryParams.getOptionalInt("blue") != null)
        {
            int r = primaryParams.getRequiredInt("red");
            int g = primaryParams.getRequiredInt("green");
            int b = primaryParams.getRequiredInt("blue");

            color = new Color(r,g,b);
        }
    }
    @Override
    public void init(Map<String, Object> context) {
        // not required
    }
}