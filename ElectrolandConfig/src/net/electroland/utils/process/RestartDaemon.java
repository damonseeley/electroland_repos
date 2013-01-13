package net.electroland.utils.process;

import java.awt.BorderLayout;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import javax.swing.JCheckBox;
import javax.swing.JFrame;

@SuppressWarnings("serial")
public class RestartDaemon extends JFrame implements ProcessExitedListener, WindowListener {

    private ProcessItem running;
    private JCheckBox state;
    private String rootDir, batFileName;
    private long pollRate;

    /**
     * @param args
     */
    public static void main(String[] args) {

        if (args.length != 3){
            System.out.println("Usage: RestartDaemon [rootDir] [batFileName] [pollRate]");
        }else{
            RestartDaemon daemon = new RestartDaemon();

            daemon.rootDir     = args[0];
            daemon.batFileName = args[1];
            daemon.pollRate    = Long.parseLong(args[2]);

            daemon.setTitle(daemon.rootDir + " " + daemon.batFileName);
            daemon.state = new JCheckBox("Restart if process dies?");
            daemon.state.setSelected(true);
            daemon.setLayout(new BorderLayout());
            daemon.getContentPane().add(daemon.state, BorderLayout.CENTER);
            daemon.setSize(250,100);
            daemon.setVisible(true);
            daemon.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            daemon.addWindowListener(daemon);
            daemon.run();
        }
    }

    public void run(){
        running = ProcessUtil.run(readBat(batFileName, 
                                  rootDir), new File(rootDir), this, pollRate);
    }

    public static String readBat(String filename, String rootDir){
        if (filename == null){
            return null;
        }else{
            File bat = new File(rootDir, filename);
            BufferedReader br;
            try {
                br = new BufferedReader(new FileReader(bat));
                String command = br.readLine();
                br.close();
                return command;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    @Override
    public void exited(ProcessItem ded) {
        if (state.isSelected()){
            run();
        }
    }

    @Override
    public void windowClosing(WindowEvent arg0) {
        state.setSelected(false);
        System.out.println("killing the process.");
        ProcessUtil.kill(running);
        System.out.println("process ded.");
    }

    @Override
    public void windowClosed(WindowEvent arg0) {}

    @Override
    public void windowActivated(WindowEvent arg0) {}

    @Override
    public void windowDeactivated(WindowEvent arg0) {}

    @Override
    public void windowDeiconified(WindowEvent arg0) {}

    @Override
    public void windowIconified(WindowEvent arg0) {}

    @Override
    public void windowOpened(WindowEvent arg0) {}
}