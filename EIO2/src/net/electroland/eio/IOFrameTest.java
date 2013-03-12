package net.electroland.eio;

import java.awt.Color;
import java.awt.Graphics;
import java.util.Map;

import javax.swing.JFrame;

import net.electroland.eio.devices.InputChannel;

@SuppressWarnings("serial")
public class IOFrameTest extends JFrame {

    final static int HEIGHT     = 400;
    final static int WIDTH      = 800;
    final static int BAR_WIDTH  = 30;
    final static int BASELINE   = HEIGHT - 50;
    final static int MAX_HEIGHT = HEIGHT - 100;

    IOManager manager;

    public IOFrameTest(IOManager manager){
        this.manager = manager;
    }

    public static void main(String[] args) {

        String propsFilename = "io.properties";
        int fps = 33;

        if (args.length > 1){
            fps = Integer.parseInt(args[0]);
        } else if (args.length > 2) {
            propsFilename = args[1];
        } else if (args.length > 3){
            System.out.println("Usage: IOFrameTest [fps] [propsfilename]");
            System.exit(-1);
        }
        System.out.println("running " + propsFilename + " at " + fps + " fps.");
        IOManager ioMgr = new IOManager();
        ioMgr.load(propsFilename);

        IOFrameTest t = new IOFrameTest(ioMgr);
        t.setSize(HEIGHT, WIDTH);
        t.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        t.setVisible(true);

        while (true){
            t.repaint();
            try {
                Thread.sleep(1000/fps);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public void paint(Graphics g){

        g.setColor(Color.BLACK);
        g.fillRect(0, 0, this.getWidth(), this.getHeight());
        g.setColor(Color.WHITE);

        // read input
        Map<InputChannel, Object> readVals = manager.read();

        // plot inputs
        for (InputChannel channel : manager.getInputChannels()){
            Object val = readVals.get(channel);
            if (val instanceof Short){

                int barHeight = scale((Short)val, MAX_HEIGHT);
                g.fillRect((int)channel.getLocation().getX(), BASELINE - barHeight, BAR_WIDTH, barHeight);
            }
            // TODO: show actual values and channel ID on screen as text
            // pause button (screen lock)
        }
    }

    public int scale(short value, int dim){
        float percent = value / (float)Short.MAX_VALUE;
        return (int)(percent * dim);
    }
}