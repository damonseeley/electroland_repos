package net.electroland.util.tsv;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * returns a JSON or Map of each line of a TSV file, one line at a time.
 * 
 * @author bradley
 *
 */
public class TSV {

    private List<String>headers;
    private BufferedReader br;

    /**
     * If the first row of the data contains the headers.
     * 
     * @param f
     * @throws IOException
     */
    public TSV(File f) throws IOException {
        this.br = new BufferedReader(new FileReader(f));
        this.headers = getHeaders(br);
    }

    /** 
     * If you want to pass the headers in as an ordered List. Assumes all lines
     * in the file are data.
     * 
     * @param f
     * @param headers
     * @throws IOException
     */
    public TSV(File f, List<String>headers) throws IOException {
        this.br = new BufferedReader(new FileReader(f));
        this.headers = headers;
    }

    /**
     * Get the next row as a Map
     * 
     * @return
     * @throws IOException
     */
    public Map<String,String> nextRow() throws IOException {

        Map<String, String>result = new HashMap<String, String>();
        int index = 0;

        String line = br.readLine();
        String[] vals= line.split("\t");

        for (String value : vals) {
            result.put(headers.get(index++), clean(value));
        }

        return result;
    }

    /**
     * Get the next row as a JSON.
     * 
     * @return
     * @throws IOException
     */
    public String nextRowJSON() throws IOException {

        Map<String, String>row = nextRow();

        StringBuffer sb = new StringBuffer();

        sb.append('{');

        for (String key : row.keySet()){
            sb.append('"').append(key).append('"')
                .append(':')
                .append('"').append(row.get(key)).append('"')
                .append(',');
        }

        if (row.keySet().size() > 0){ // removing potential training comma
            sb.setLength(sb.length() - 1);
        }

        sb.append('}');

        return sb.toString();
    }

    /**
     * Determine if there are any remaining lines to read.
     * 
     * @return
     * @throws IOException
     */
    public boolean ready() throws IOException {
        return br.ready();
    }

    /**
     * Close the backing file.
     * 
     * @throws IOException
     */
    public void close() throws IOException {
        br.close();
    }

    private List<String> getHeaders(BufferedReader br) throws IOException{

        ArrayList<String>headers = new ArrayList<String>();

        String line = br.readLine();
        String[] keys= line.split("\t");

        for (String key : keys) {
            headers.add(clean(key));
        }

        return headers;
    }

    private String clean(String token){

        token.trim();

        if (token.startsWith("\"") && token.endsWith("\"") ||
            token.startsWith("'")  && token.endsWith("'")) {
            return token.substring(1, token.length() - 2);
        } else {
            return token;
        }
    }
}