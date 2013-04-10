package net.electroland.eio;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;

import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;


@SuppressWarnings("serial")
public class IOFrameTest extends JFrame implements IOListener, ActionListener {

    public static final String RECORD = "Start recording";
    public static final String STOP   = "Stop and save";
    private EIOManager manager;
    private ValueSet lastRead = new ValueSet();
    private Collection<ValueSet> recording;
    final JFileChooser fc = new JFileChooser();

    public IOFrameTest(EIOManager manager){
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
        EIOManager ioMgr = new EIOManager();
        ioMgr.load(propsFilename);

        IOFrameTest t = new IOFrameTest(ioMgr);
        t.setSize(1200, 600);
        t.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        JButton record = new JButton(RECORD);
        record.addActionListener(t);
        record.setPreferredSize(new Dimension(100,20));
        t.setLayout(new FlowLayout(FlowLayout.RIGHT));
        t.getContentPane().add(record);

        t.setVisible(true);


        ioMgr.addListener(t);
        ioMgr.start();
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
        g.fillRect(0, 50, this.getWidth(), this.getHeight());

        // draw graph baseline
        g.setColor(Color.WHITE);
        g.drawLine(0, baseline, width, baseline);

        // plot inputs
        for (InputChannel channel : manager.getInputChannels()){

            Value val = lastRead.get(channel.id);
            int left  = (int)channel.getLocation().getX();
            int top   = baseline;

            if (val != null && isRecent(lastRead.getReadTime())){
                // cast based on val.
                int barHeight   = scale(val.getFilteredValue(), maxBarHite);
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
                g2d.drawString("" + val.getFilteredValue(), left, top);
            }
            // id
            g2d.setColor(Color.WHITE);
            g2d.drawString(channel.getId(), left, baseline+10);
        }
    }

    public int scale(int value, int dim){
        float percent = value / (float)Short.MAX_VALUE;
        return (int)(percent * dim);
    }

    public boolean isRecent(long time){
        return System.currentTimeMillis() - time < 100;
    }

    @Override
    public void dataReceived(ValueSet incoming) {
        lastRead = incoming;
        if (recording != null){
            recording.add(incoming);
        }
        repaint();
    }

    @Override
    public void actionPerformed(ActionEvent evt) {

        JButton button = (JButton)evt.getSource();
        if (RECORD.equals(button.getText())){
            recording = new ArrayList<ValueSet>();
            System.out.println("starting recording");
            button.setText(STOP);
        } else {
            System.out.println("saving recording");
            button.setText(RECORD);
            save(recording);
        }
        repaint();
    }

    public void save(Collection<ValueSet> recording){

        this.recording = null; // stops recording

        if (fc.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {

            try {

                File file       = fc.getSelectedFile();
                PrintWriter pw  = new PrintWriter(file);
                StringBuffer sb = new StringBuffer();

                for (ValueSet v : recording){
                    v.serialize(sb);
                    pw.println(sb.toString());
                    sb.setLength(0);
                }

                pw.flush();
                pw.close();

            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
    }
}