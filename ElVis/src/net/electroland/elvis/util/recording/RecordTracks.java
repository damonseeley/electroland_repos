package net.electroland.elvis.util.recording;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.net.SocketException;

import net.electroland.elvis.blobtracking.BaseTrack;
import net.electroland.elvis.blobtracking.TrackResults;
import net.electroland.elvis.net.TrackUDPClient;

public class RecordTracks extends TrackUDPClient {

    private BufferedWriter output;
    private StringBuilder buffer;

    public static void main(String args[]) throws Exception {
        if (args.length != 2){
            System.out.println("Usage: RecordPresenceGrid [port] [filename]");
        }else{
            RecordTracks r = new RecordTracks(new Integer(args[0]));
            r.output = new BufferedWriter(new FileWriter(new File(args[1])));
            r.buffer = new StringBuilder();
        }
    }

    public RecordTracks(int port) throws SocketException {
        super(port);
    }

    @Override
    public void handle(TrackResults<BaseTrack> t) {
        if (output != null){

            buffer.setLength(0);
            buffer.append(System.currentTimeMillis());
            buffer.append(':');
            t.buildString(buffer);

            try {

                output.write(buffer.toString());
                output.newLine();
                output.flush();

            } catch (IOException e) {
                e.printStackTrace();
                System.exit(-1);
            }finally{

                try {
                    output.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}