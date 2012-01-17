package net.electroland.ea.content;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.util.Map;

import net.electroland.ea.Content;
import net.electroland.utils.ParameterMap;

public class SolidColorContent extends Content {

    public Color getColor() {
        return color;
    }
    public void setColor(Color color) {
        this.color = color;
    }

    private Color color;

    public SolidColorContent(Color c){
        if (c==null){
            System.out.println("NO COLOR!");
        }
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
        // not required, but might be cool to parse differen color
        // specifications like $hex=#ffffff or $red=255, $blue=255, $green=255
    }

    @Override
    public void init(Map<String, Object> context) {
        // not required
    }

}
