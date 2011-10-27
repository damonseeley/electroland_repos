package net.electroland.eio.tools;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Rectangle;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;

import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.vecmath.Point3d;

import net.electroland.eio.IOManager;
import net.electroland.eio.IOState;
import net.electroland.eio.IState;

@SuppressWarnings("serial")
public class IStatePanel extends JComponent implements MouseListener {

    IOManager iom;
    double scale;
    int side;

    public IStatePanel(IOManager iom, double scale, int side){
        this.iom = iom;
        this.scale = scale;
        this.side = side;
    }

    public static void main(String args[])
    {
        // filename, scale
        if (args.length != 3){
            System.out.println("Usage: IStatePanel [io props filename] [scale] [sqare size]");
            System.exit(0);
        }

        String filename = args[0];
        Double scale = Double.valueOf(args[1]);
        System.out.println("will scale sensors to " + scale + " for rendering.");
        int side = Integer.valueOf(args[2]);
        System.out.println("will draw each sensor at " + side + " pixels on a side.");

        IOManager iom = new IOManager();
        iom.load(filename);

        JFrame frame = new JFrame(filename + " at " + scale + " scale, " + side + " on a side.");
        IStatePanel display = new IStatePanel(iom, scale, side);
        frame.getContentPane().add(display);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(1280,300);
        frame.setVisible(true);
        display.addMouseListener(display);

        iom.start();
        while (true)
        {
            display.repaint();
            try {
                Thread.sleep(33);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    public void paint(Graphics g){
        g.setColor(Color.BLACK);
        g.fillRect(0, 0, this.getWidth(), this.getHeight());

        for (IOState state : iom.getStates())
        {
            Point3d l = state.getLocation();
            if (((IState)state).getState()){
                g.setColor(Color.WHITE);
                //g.setColor(Color.RED);
                g.fillRect((int)(l.x * scale), (int)(l.y * scale),
                                                side, side);
            }else{
                g.setColor(Color.GRAY);
                //g.fillRect((int)(l.x * scale), (int)(l.y * scale), side, side);
                g.drawRect((int)(l.x * scale), (int)(l.y * scale), side, side);
            }
        }
    }

    @Override
    public void mouseReleased(MouseEvent evt) {
        for (IOState state : iom.getStates())
        {
            Point3d l = state.getLocation();
            Rectangle r = new Rectangle((int)(l.x * scale), (int)(l.y * scale),
                                        side, side);
            if (r.contains(evt.getPoint())){
                System.out.println(state);
            }
        }
    }

    @Override
    public void mouseClicked(MouseEvent arg0) {
    }

    @Override
    public void mouseEntered(MouseEvent arg0) {
    }

    @Override
    public void mouseExited(MouseEvent arg0) {
    }

    @Override
    public void mousePressed(MouseEvent arg0) {
    }

}