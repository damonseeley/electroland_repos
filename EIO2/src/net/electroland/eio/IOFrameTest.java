package net.electroland.eio;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.util.Map;

import javax.swing.JFrame;


@SuppressWarnings("serial")
public class IOFrameTest extends JFrame {

    private IOManager manager;

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
        t.setSize(1200, 600);
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

        Graphics2D g2d = (Graphics2D)g;

        int height      = this.getHeight();
        int width       = this.getWidth();
        int barWidth       = 10;
        int baseline    = height / 2;
        int margin      = 50;
        int maxBarHite  = baseline - margin;
        Color barColor = new Color(255,255,255,150);

        g.setColor(Color.BLACK);
        g.fillRect(0, 0, this.getWidth(), this.getHeight());

        // read input
        Map<InputChannel, Value> readVals = manager.read();

        // draw graph baseline
        g.setColor(Color.WHITE);
        g.drawLine(0, baseline, width, baseline);

        // plot inputs
        for (InputChannel channel : manager.getInputChannels()){
            Value val = readVals.get(channel);

            // cast based on val.
            int barHeight   = scale(val.getValue(), maxBarHite);
            int left        = (int)channel.getLocation().getX();
            int top = baseline;
            if (barHeight > 0){
                top -= barHeight;
            }

            // bar
            g2d.setColor(barColor);
            g2d.fillRect(left, top, barWidth, barHeight);

            Font font = new Font("Arial", Font.PLAIN, 9);
            g2d.setFont(font);

            // value
            g2d.setColor(Color.WHITE);
            g2d.drawString("" + val.getValue(), left, top);

            // id
            g2d.setColor(Color.WHITE);
            g2d.drawString(channel.getId(), left, baseline+10);
        }
    }

    public int scale(int value, int dim){
        float percent = value / (float)Short.MAX_VALUE;
        return (int)(percent * dim);
    }
}