package net.electroland.elvis.util.recording;

import java.io.IOException;
import java.net.SocketException;

import net.electroland.elvis.net.GridData;
import net.electroland.elvis.net.PresenceGridUDPClient;

public class RecordPresenceGrid extends PresenceGridUDPClient implements Shutdownable {

    private Recorder recorder;

    public RecordPresenceGrid(int port) throws SocketException {
        super(port);
    }

    public static void main(String args[]) throws Exception {
        if (args.length != 2){
            System.out.println("Usage: RecordPresenceGrid [port] [filename]");
            System.exit(-1);
        }else{
            RecordPresenceGrid r = new RecordPresenceGrid(new Integer(args[0]));
            Runtime.getRuntime().addShutdownHook(new ShutdownThread(r));
            r.recorder = new FileRecorder(args[1], "Recorded from " + r.getClass().getName());
            r.start();
        }
    }

    @Override
    public void handle(GridData t) {
        if (recorder != null){
            try {
                recorder.recordLine(t);
            } catch (IOException e) {
                e.printStackTrace();
                shutdown();
            }
        }
    }

    @Override
    public void shutdown(){
        System.out.println("Shutting down...");
        recorder.close();
        this.stopRunning();
    }
}