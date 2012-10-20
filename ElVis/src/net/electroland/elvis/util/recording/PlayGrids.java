package net.electroland.elvis.util.recording;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.List;

import net.electroland.elvis.net.GridData;
import net.electroland.elvis.net.UDPBroadcaster;

public class PlayGrids extends UDPBroadcaster{

    public PlayGrids(int port) throws SocketException,
            UnknownHostException {
        super("localhost", port);
        this.start();
    }

    public PlayGrids(String address, int port) throws SocketException,
            UnknownHostException {
        super(address, port);
        this.start();
    }

    /**
     * @param args
     */
    public static void main(String[] args) {

        if (args.length != 2){
            System.out.println("Usage: RecordPresenceGrid [ip:port] [filename]");
            System.exit(-1);

        }else{

            try{
                List<GridDataPoint> samples = readGridsFromFile(args[1]);

                int port = getPort(args[0]);
                String host = getHost(args[0]);

                PlayGrids pg;
                if (host != null){
                    pg = new PlayGrids(host, port);
                }else{
                    pg = new PlayGrids(port);
                }

                while (true){
                    long last = samples.get(0).time; // slight bug: first sample has zero sleep.
                    for (GridDataPoint sample : samples){
                        pg.send(sample.data);
                        Thread.sleep(sample.time - last);
                        last = sample.time;
                    }
                }

            }catch(IOException e){
                e.printStackTrace(System.err);
                System.exit(-1);
            } catch (InterruptedException e) {
                e.printStackTrace(System.err);
                System.exit(-1);
            }
        }
    }

    public static String getHost(String host) {
        int i = host.indexOf(':');
        return i == -1 ? null : host.substring(0, i);
    }

    public static int getPort(String host) {
        int i = host.indexOf(':');
        return i == -1 ? new Integer(host) : new Integer(host.substring(i + 1, host.length()));
    }

    public static List<GridDataPoint> readGridsFromFile(String filename) throws IOException{
        ArrayList<GridDataPoint> a = new ArrayList<GridDataPoint>();
        BufferedReader br = new BufferedReader(new FileReader(new File(filename)));
        br.readLine(); // consume header
        while (br.ready()){
            a.add(new GridDataPoint(br.readLine()));
        }
        br.close();
        return a;
    }
}

class GridDataPoint {

    long time;
    GridData data;

    public GridDataPoint(String dataStr){
        int i = dataStr.indexOf(':');
        time = new Long(dataStr.substring(0, i));
        data = new GridData(dataStr.substring(i + 1, dataStr.length()));
    }
}