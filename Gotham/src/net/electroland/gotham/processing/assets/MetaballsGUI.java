package net.electroland.gotham.processing.assets;

import java.awt.Rectangle;

import net.electroland.utils.ElectrolandProperties;
import processing.core.PApplet;
import controlP5.Button;
import controlP5.ControlEvent;
import controlP5.ControlListener;
import controlP5.ControlP5;
import controlP5.ControlWindow;
import controlP5.Controller;
import controlP5.Knob;
import controlP5.Slider;
import controlP5.Textfield;
import controlP5.Toggle;

/** This is ONLY used by Metaballs3 which is NOT the production version.  For the UI
 * for the production version, please see MetaballsProps.java
 * 
 * @author damon
 *
 */
public class MetaballsGUI implements ControlListener {

    private ControlP5 control;
    private ControlWindow window;

    public Controller<Knob> r1, r2, r3, bgR;
    public Controller<Knob> g1, g2, g3, bgG;
    public Controller<Knob> b1, b2, b3, bgB;

    private Controller<Slider> cohesiveness, repelVelocity, repelForce, threshold;
    private Controller<Button> dump;

    private Controller<Textfield> gridXinset, gridYinset, gridWidth, gridHeight;
    private Controller<Textfield> gridCanvasXinset, gridCanvasYinset, gridCanvasWidth, gridCanvasHeight;
    private Controller<Toggle> showGrid;
    
    private Rectangle grid, canvas;

    public MetaballsGUI(PApplet p) {

        ElectrolandProperties ep = new ElectrolandProperties("Gotham-global.properties");

        grid = new Rectangle(ep.getRequiredInt("lava", "grid", "xinset"),
                ep.getRequiredInt("lava", "grid", "yinset"),
                ep.getRequiredInt("lava", "grid", "width"),
                ep.getRequiredInt("lava", "grid", "height"));
        canvas = new Rectangle(ep.getRequiredInt("lava", "gridOnCanvas", "xinset"),
                ep.getRequiredInt("lava", "gridOnCanvas", "yinset"),
                ep.getRequiredInt("lava", "gridOnCanvas", "width"),
                ep.getRequiredInt("lava", "gridOnCanvas", "height"));

        control = new ControlP5(p);
        window = control
                .addControlWindow("Lava_Control_Window", 100, 100, 600, 450)
                .hideCoordinates().setBackground(0);

        r1 = control.addKnob("red 1")
                    .setRange(0, 255)
                    .setValue(ep.getRequiredDouble("lava", "color1", "r").floatValue())
                    .setPosition(10, 10)
                    .setRadius(30);
        g1 = control.addKnob("green 1")
                    .setRange(0, 255)
                    .setValue(ep.getRequiredDouble("lava", "color1", "g").floatValue())
                    .setPosition(10, 90)
                    .setRadius(30);
        b1 = control.addKnob("blue 1")
                    .setRange(0, 255)
                    .setValue(ep.getRequiredDouble("lava", "color1", "b").floatValue())
                    .setPosition(10, 170)
                    .setRadius(30);
        r1.moveTo(window);
        g1.moveTo(window);
        b1.moveTo(window);

        r2 = control.addKnob("red 2")
                    .setRange(0, 255)
                    .setValue(ep.getRequiredDouble("lava", "color2", "r").floatValue())
                    .setPosition(90, 10)
                    .setRadius(30);
        g2 = control.addKnob("green 2")
                    .setRange(0, 255)
                    .setValue(ep.getRequiredDouble("lava", "color2", "g").floatValue())
                    .setPosition(90, 90)
                    .setRadius(30);
        b2 = control.addKnob("blue 2")
                    .setRange(0, 255)
                    .setValue(ep.getRequiredDouble("lava", "color2", "b").floatValue())
                    .setPosition(90, 170)
                    .setRadius(30);
        r2.moveTo(window);
        g2.moveTo(window);
        b2.moveTo(window);

        r3 = control.addKnob("red 3")
                    .setRange(0, 255)
                    .setValue(ep.getRequiredDouble("lava", "color3", "r").floatValue())
                    .setPosition(170, 10)
                    .setRadius(30);
        g3 = control.addKnob("green 3")
                    .setRange(0, 255)
                    .setValue(ep.getRequiredDouble("lava", "color3", "g").floatValue())
                    .setPosition(170, 90)
                    .setRadius(30);
        b3 = control.addKnob("blue 3")
                    .setRange(0, 255)
                    .setValue(ep.getRequiredDouble("lava", "color3", "b").floatValue())
                    .setPosition(170, 170)
                    .setRadius(30);
        r3.moveTo(window);
        g3.moveTo(window);
        b3.moveTo(window);

        bgR = control.addKnob("background red")
                    .setRange(0, 255)
                    .setValue(ep.getRequiredDouble("lava", "background", "r").floatValue())
                    .setPosition(250, 10)
                    .setRadius(30);
        bgG = control.addKnob("background green")
                    .setRange(0, 255)
                    .setValue(ep.getRequiredDouble("lava", "background", "g").floatValue())
                    .setPosition(250, 90)
                    .setRadius(30);
        bgB = control.addKnob("background blue")
                    .setRange(0, 255)
                    .setValue(ep.getRequiredDouble("lava", "background", "b").floatValue())
                    .setPosition(250, 170)
                    .setRadius(30);
        bgR.moveTo(window);
        bgG.moveTo(window);
        bgB.moveTo(window);

        cohesiveness = control.addSlider("cohesiveness")
                    .setRange(.005f, .01f)
                    .setValue(ep.getRequiredDouble("lava", "physics", "cohesivness").floatValue())
                    .setPosition(10, 270)
                    .setWidth(250);
        cohesiveness.moveTo(window);

        repelVelocity = control.addSlider("repelVelocity")
                    .setRange(10, 1000) 
                    .setValue(ep.getRequiredDouble("lava", "physics", "maxVelocity").floatValue())
                    .setPosition(10, 300)
                    .setWidth(250);
        repelVelocity.moveTo(window);

        repelForce = control.addSlider("repelForce")
                    .setRange(10, 1000)
                    .setValue(ep.getRequiredDouble("lava", "physics", "repellForce").floatValue())
                    .setPosition(10, 330)
                    .setWidth(250);
        repelForce.moveTo(window);

        threshold = control.addSlider("threshold")
                    .setRange(0, 100)
                    .setValue(ep.getRequiredDouble("lava", "physics", "threshold").floatValue())
                    .setPosition(10, 360)
                    .setWidth(250);
        threshold.moveTo(window);

        dump = control.addButton("console dump")
                    .setPosition(10, 390);
        dump.addListener(this);
        dump.moveTo(window);

        gridXinset = control.addTextfield("grid X inset").setText(ep.getRequiredInt("lava", "grid", "xinset").toString()).setPosition(400, 10).setWidth(90);
        gridYinset = control.addTextfield("grid Y inset").setText(ep.getRequiredInt("lava", "grid", "yinset").toString()).setPosition(500, 10).setWidth(90);
        gridWidth = control.addTextfield("grid width").setText(ep.getRequiredInt("lava", "grid", "width").toString()).setPosition(400, 50).setWidth(90);
        gridHeight = control.addTextfield("grid height").setText(ep.getRequiredInt("lava", "grid", "height").toString()).setPosition(500, 50).setWidth(90);
        gridXinset.moveTo(window);
        gridYinset.moveTo(window);
        gridWidth.moveTo(window);
        gridHeight.moveTo(window);

        gridCanvasXinset = control.addTextfield("grid canvas X inset").setText("80").setPosition(400, 110).setWidth(90);
        gridCanvasYinset = control.addTextfield("grid canvas Y inset").setText("70").setPosition(500, 110).setWidth(90);
        gridCanvasWidth = control.addTextfield("grid canvas width").setText("540").setPosition(400, 150).setWidth(90);
        gridCanvasHeight = control.addTextfield("grid canvas height").setText("364").setPosition(500, 150).setWidth(90);
        gridCanvasXinset.moveTo(window);
        gridCanvasYinset.moveTo(window);
        gridCanvasWidth.moveTo(window);
        gridCanvasHeight.moveTo(window);

        showGrid = control.addToggle("show grid").setState(ep.getRequiredBoolean("lava", "gridOnCanvas", "debug")).setPosition(400, 200);
        showGrid.moveTo(window);
    }

    public Color getColor1(){
        return new Color(r1.getValue(), g1.getValue(), b1.getValue());
    }
    public Color getColor2(){
        return new Color(r2.getValue(), g2.getValue(), b2.getValue());
    }
    public Color getColor3(){
        return new Color(r3.getValue(), g3.getValue(), b3.getValue());
    }
    public Color getBGColor(){
        return new Color(bgR.getValue(), bgG.getValue(), bgB.getValue());
    }
    public float getCohesiveness(){
        return cohesiveness.getValue();
    }
    public float getRepelVelocity(){
        return repelVelocity.getValue();
    }
    public float getRepelForce(){
        return repelForce.getValue();
    }
    public float getThreshold(){
        return threshold.getValue();
    }

    public Rectangle getGrid(){

        // parse inputs
        int x = getVal(gridXinset, grid.x);
        int y = getVal(gridYinset, grid.y);
        int w = getVal(gridWidth, grid.width);
        int h = getVal(gridHeight, grid.height);

        if (x + w > 80){
            System.err.println("illegal inset/width");
            return grid;
        }
        if (y + h > 40){
            System.err.println("illegal inset/height");
            return grid;
        }
        // write back if anything failed
        grid = new Rectangle(x,y,w,h);

        return grid;
    }

    public Rectangle getGridCanvas(){
        // parse inputs
        int x = getVal(gridCanvasXinset, grid.x);
        int y = getVal(gridCanvasYinset, grid.y);
        int w = getVal(gridCanvasWidth, grid.width);
        int h = getVal(gridCanvasHeight, grid.height);

        // write back if anything failed
        canvas = new Rectangle(x,y,w,h);

        return canvas;
    }

    public boolean showGrid(){
        return ((Toggle)showGrid).getState();
    }

    public static int getVal(Controller<Textfield> input, int fail){
        try{
            Integer i = Integer.parseInt(((Textfield)input).getText());
            return i.intValue();
        }catch(NumberFormatException e){
            return fail;
        }
    }

    @Override
    public void controlEvent(ControlEvent arg0) {
        System.out.println();
        System.out.println("current values:");
        System.out.println("===============");
        System.out.println("color1:           (" + r1.getValue() + ", " + g1.getValue() + ", " + b1.getValue() + ")");
        System.out.println("color2:           (" + r2.getValue() + ", " + g2.getValue() + ", " + b2.getValue() + ")");
        System.out.println("color3:           (" + r3.getValue() + ", " + g3.getValue() + ", " + b3.getValue() + ")");
        System.out.println("bg color:         (" + bgR.getValue() + ", " + bgG.getValue() + ", " + bgB.getValue() + ")");
        System.out.println("cohesiveness:     " + cohesiveness.getValue());
        System.out.println("repell velocity:  " + repelVelocity.getValue());
        System.out.println("repell force:     " + repelForce.getValue());
        System.out.println("threshold:        " + threshold.getValue());
    }
}