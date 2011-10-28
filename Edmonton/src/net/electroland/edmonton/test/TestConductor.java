package net.electroland.edmonton.test;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.awt.image.BufferedImage;
import java.io.IOException;

import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.vecmath.Point3d;

import net.electroland.eio.IOManager;
import net.electroland.eio.IOState;
import net.electroland.eio.IState;
import net.electroland.utils.OptionException;
import net.electroland.utils.lighting.ELUManager;
import net.electroland.utils.lighting.Fixture;
import net.electroland.utils.lighting.InvalidPixelGrabException;
import net.electroland.utils.lighting.canvas.ELUCanvas2D;

@SuppressWarnings("serial")
public class TestConductor extends JComponent implements MouseMotionListener{

    private ELUManager elu;
    private IOManager eio;
    private TestModel model;

    // width to render sensors & lights
    final static int side = 3;
    final static int lightside = side*4;
    final static int dbrightness = 10;
    final static long delay = 1000;

    public TestConductor()
    {
        elu = new ELUManager();
        eio = new IOManager();
        try {
            elu.load("EIA-ELU.properties");
            eio.load("EIA-EIO.properties");
            eio.start();
            model = new TestModel(eio.getStates(), dbrightness);
        } catch (OptionException e) {
            e.printStackTrace();
            System.exit(0);
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(0);
        }
    }

    public static void main(String args[])
    {
        JFrame frame = new JFrame();
        TestConductor display = new TestConductor();
        frame.getContentPane().add(display);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(700,50);
        frame.setVisible(true);
        display.addMouseMotionListener(display);

        while(true){
            display.repaint();
            try {
                Thread.sleep(delay);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    int pixels[];

    public void paint(Graphics g)
    {

        // background of JComponent
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, this.getWidth(), this.getHeight());

        // canvas
        ELUCanvas2D c = (ELUCanvas2D)elu.getCanvas("EIAspan");
        int width = c.getDimensions().width;
        int height = c.getDimensions().height;

        // buffered image to draw on.  This will get synced with lights.
        BufferedImage b = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        // paing it black
        Graphics bg = b.getGraphics();
        bg.setColor(Color.BLACK);
        bg.fillRect(0,0, width, height);

        // sync the model
        model.update();

        // render sprites on buffered image
        boolean d = true;
        for (IOState state : eio.getStates())
        {
            Point3d l = state.getLocation();
            // render sprite
            int brightness = model.getBrightness(state);
            if (d){
                d = false;
            }
            bg.setColor(new Color(brightness, brightness, brightness));
            bg.fillRect((int)(l.x)-(lightside/2), (int)(l.y)-(lightside/2),
                    lightside, lightside);
        }
        
        // if mouse is dragged
        if (mouseOn){
            bg.setColor(Color.WHITE);
            bg.fillRect((int)(mouseX)-(lightside/2), 0, lightside, height);
            
        }

        // sync lights
        if (pixels == null){
            pixels = new int[width * height];
        }
        b.getRGB(0, 0, width, height, pixels, 0, width);
        try {
            c.sync(pixels);
            elu.syncAllLights();
        } catch (InvalidPixelGrabException e) {
            e.printStackTrace();
        }

        // paint canvas onto JPanel
        g.drawImage(b, 0, 0, this);

        // render sensors
        for (IOState state : eio.getStates())
        {
            Point3d l = state.getLocation();

            if (((IState)state).getState()){
                g.setColor(Color.RED);
                g.fillRect((int)(l.x), (int)(l.y), side, side);
            }else{
                g.setColor(Color.WHITE);
                g.fillRect((int)(l.x), (int)(l.y), side, side);
            }
        }
        // render fixtures
        for (Fixture fix : elu.getFixtures())
        {
            Point3d l = fix.getLocation();
            g.setColor(Color.BLUE);
            g.fillRect((int)(l.x), (int)(l.y), side, side);
        }
    }
    boolean mouseOn;
    int mouseX;
    @Override
    public void mouseDragged(MouseEvent arg0) {
        mouseOn = true;
        mouseX = arg0.getX();
    }

    @Override
    public void mouseMoved(MouseEvent arg0) {
        mouseOn = false;
    }
}