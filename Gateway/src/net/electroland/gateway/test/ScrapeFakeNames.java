package net.electroland.gateway.test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.URL;
import java.net.URLConnection;

/**
 * grabs a list of 6000 fakes names from kleimo.com
 * @author bradley
 *
 */
public class ScrapeFakeNames {

    public static void main(String args[]){
        try {
            File f = new File(args[0]);
            PrintWriter pw = new PrintWriter(new FileWriter(f));

            String data = "type=3&number=30&obscurity=20&Go=Generate%20Random%20Name(s)";

            for (int i = 0; i < 6000 / 30; i++){

                URL url = new URL("http://www.kleimo.com/random/name.cfm");
                URLConnection conn = url.openConnection();
                conn.setDoOutput(true);
                OutputStreamWriter wr = new OutputStreamWriter(conn.getOutputStream());
                wr.write(data);
                wr.flush();

                BufferedReader rd = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                String line;
                while (rd.ready() && (line = rd.readLine().trim()) != null) {

                    if (line.length() > 0){
                        int ascii = (byte)line.charAt(0);
                        if (ascii > 48 && ascii < 58){
                            int start = line.indexOf(';') + 2;
                            pw.println(line.substring(start, line.length() - 4));
                            System.out.println(line.substring(start, line.length() - 4));
                        }
                    }
                }
                pw.flush();
                wr.close();
                rd.close();
                Thread.sleep(1000);
            }
            pw.close();
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }
    }
}