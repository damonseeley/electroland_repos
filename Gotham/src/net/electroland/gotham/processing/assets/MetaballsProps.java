package net.electroland.gotham.processing.assets;

import java.awt.Point;
import java.awt.Rectangle;

import net.electroland.utils.ElectrolandProperties;
import processing.core.PApplet;
import controlP5.Button;
import controlP5.ControlEvent;
import controlP5.ControlListener;
import controlP5.ControlP5;
import controlP5.ControlWindow;
import controlP5.Controller;
import controlP5.ControllerInterface;
import controlP5.Slider;
import controlP5.Textfield;
import controlP5.Toggle;

public class MetaballsProps implements ControlListener {

    // DO NOT CHANGE ANY OF THESE: HASH KEYS!
    final static public String COHESIVENESS            = "cohesiveness";
    final static public String REPELL_FORCE            = "repellForce";
    final static public String REPELL_VELOCITY_CEILING = "repellVelocityCeiling";
    final static public String THRESHOLD               = "threshold";
    final static public String FRICTION                = "friction";
    final static public String MIN_VELOCITY            = "minVelocity";
    final static public String MAX_VELOCITY            = "maxVelocity";
    final static public String BALL_SCALE              = "ballScale";
    final static public String BALL_OPACITY            = "ballOpacity";
    final static public String PRESENCE_RADIUS         = "presenceRadius";
    final static public String PRESENCE_OPACITY        = "presenceOpacity";
    final static public String BLUR                    = "blur";
    final static public String GRID                    = "grid";
    final static public String GRID_ON_CANVAS          = "gridOnCanvas";
    final static public String ENABLE_GRID             = "enableGrid";
    final static public String SHOW_GRID               = "showGrid";

    private String wallName;
    private ControlP5 p5;
    private ControlWindow window;
    private Point placement = new Point(10, 20);

    public MetaballsProps(PApplet parent, String wallName, ElectrolandProperties props){

            this.wallName = wallName;

            p5 = new ControlP5(parent);

            window = p5
                     .addControlWindow(wallName + " controller", 100, 100, 600, 500)
                     .hideCoordinates().setBackground(0);

            addSlider(COHESIVENESS,            props);
            addSlider(REPELL_FORCE,            props);
            addSlider(REPELL_VELOCITY_CEILING, props);
            addSlider(THRESHOLD,               props);
            addSlider(FRICTION,                props);
            addSlider(MIN_VELOCITY,            props);
            addSlider(MAX_VELOCITY,            props);
            addSlider(BALL_SCALE,              props);
            addSlider(BALL_OPACITY,            props);
            addSlider(PRESENCE_RADIUS,         props);
            addSlider(PRESENCE_OPACITY,        props);
            addSlider(BLUR,                    props);

            nextColumn();

            addBoundary(GRID,                  props);
            addBoundary(GRID_ON_CANVAS,        props);

            addSwitch(ENABLE_GRID,             props);
            addSwitch(SHOW_GRID,               props);

            addConsoleOutputButton();
    }

    private void addSlider(String sliderName, ElectrolandProperties props){
        System.out.println(sliderName);
        float left  = props.getRequiredDouble("lava",   sliderName, "min").floatValue();
        float right = props.getRequiredDouble("lava",   sliderName, "max").floatValue();
        float init  = props.getRequiredDouble(wallName, sliderName, "default").floatValue();

        Controller<Slider> control = p5.addSlider(sliderName)
                                      .setRange(left, right)
                                      .setValue(init)
                                      .setPosition(placement.x, placement.y)
                                      .setWidth(window.component().getWidth() / 2 - 50);
        control.moveTo(window);

        nextRow();
    }
    public void addBoundary(String name, ElectrolandProperties props){

        int x = props.getRequiredInt(wallName, name, "x");
        int y = props.getRequiredInt(wallName, name, "y");
        int w = props.getRequiredInt(wallName, name, "width");
        int h = props.getRequiredInt(wallName, name, "height");

        // DO NOT change the label names, because we use them as hash keys.
        Controller<Textfield> xc = p5.addTextfield(name + " x inset")
                                     .setText("" + x)
                                     .setPosition(placement.x + 75, placement.y)
                                     .setWidth(90);
        Controller<Textfield> yc = p5.addTextfield(name + " y inset")
                                     .setText("" + y)
                                     .setPosition(placement.x + 175, placement.y)
                                     .setWidth(90);
        nextRow();

        // DO NOT change the label names, because we use them as hash keys.
        Controller<Textfield> wc = p5.addTextfield(name + " width")
                                     .setText("" + w)
                                     .setPosition(placement.x + 75, placement.y)
                                     .setWidth(90);
        Controller<Textfield> hc = p5.addTextfield(name + " height")
                                     .setText("" + h)
                                     .setPosition(placement.x + 175, placement.y)
                                     .setWidth(90);
        nextRow();

        xc.moveTo(window);  yc.moveTo(window);
        wc.moveTo(window);  hc.moveTo(window);
    }

    public void addSwitch(String name, ElectrolandProperties props){

        boolean initState = props.getRequiredBoolean(wallName, "toggles", name);

        Controller<Toggle> t = p5.addToggle(name)
                                 .setState(initState)
                                 .setPosition(placement.x + 75, placement.y);
        t.moveTo(window);

        nextRow();
    }

    // TODO: this needs to be based on image cycling.
    public Color getColor(int ballId){
        switch (ballId){
            case(0): // BACKGROUND
                return new Color(0, 50, 255);
            case(1): // balls.1
                return new Color(255, 0, 0);
            case(2): // balls.2
                return new Color(255, 127, 0);
            case(3): // balls.3
                return new Color(255, 0, 255);
            default:
                return null;
        }
    }

    public float getValue(String name){
        return ((Slider)p5.getController(name)).getValue();
    }

    public boolean getState(String name){
        return ((Toggle)p5.getController(name)).getState();
    }

    public Rectangle getBoundary(String name){

        try{
            int x = new Integer(((Textfield)p5.getController(name + " x inset")).getText());
            int y = new Integer(((Textfield)p5.getController(name + " y inset")).getText());
            int w = new Integer(((Textfield)p5.getController(name + " width")).getText());
            int h = new Integer(((Textfield)p5.getController(name + " height")).getText());

            return new Rectangle(x, y, w, h);
        }catch(NumberFormatException e){
            return new Rectangle(0, 0, 1, 1);
        }
    }

    private void nextRow(){
        placement.y += 40;
    }
    private void nextColumn(){
        placement.y = 20;
        placement.x = window.component().getWidth() / 2 + 10;
    }

    private void addConsoleOutputButton(){
        Controller<Button> dump = p5.addButton("console dump")
                .setPosition(placement.x + 75, placement.y);
        dump.addListener(this);
        dump.moveTo(window);
        nextRow();

    }

    @Override
    public void controlEvent(ControlEvent arg0) {
        for (ControllerInterface<?> list : p5.getAll()){
            Controller<?> c = p5.getController(list.getName());
            if (c instanceof Slider){
                System.out.println(c.getName() + "=" + c.getValue());
            } else if (c instanceof Textfield){
                System.out.println(c.getName() + "=" + ((Textfield)c).getText());
            }
        }
    }
}