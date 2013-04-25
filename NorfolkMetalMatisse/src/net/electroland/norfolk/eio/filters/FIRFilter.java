package net.electroland.norfolk.eio.filters;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import net.electroland.eio.Value;
import net.electroland.eio.filters.Filter;
import net.electroland.utils.ParameterMap;

public class FIRFilter implements Filter {

    private BufferedReader irDataReader;
    private float[] IR;
    private float[] buffer;
    private int writeInd = 0;
    
    @Override
    public void configure(ParameterMap params) {
        
        Integer irLenToUse = params.getOptionalInt("irLen");
        
        // Is it an OK practice in Java to actually load this from a file at runtime?
        String irDataFileName = params.getRequired("irFileName");
        try {
            irDataReader = new BufferedReader(
                               new FileReader(
                                   new File(irDataFileName)));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        
        try{
            // First line specifies the length of the response
            int irLen = Integer.parseInt( irDataReader.readLine() );
            
            // If an optional irLen was specified and it is shorter than the 
            //     IR data, only use that reduced length
            if( irLenToUse != null && irLenToUse < irLen )
                irLen = irLenToUse;
            
            IR = new float[irLen];
            buffer = new float[irLen];
            
            // Remaining lines specify the IR
            for(int i=0; i<irLen; i++) {
                IR[i] = Float.parseFloat( irDataReader.readLine() );
                buffer[i] = 0.0f;
            }
            
            irDataReader.close();
            
        } catch (IOException e) {
            e.printStackTrace();
        }
        
    }

    @Override
    public void filter(Value in) {
        
        buffer[writeInd] = in.getValue();
        
        float output = 0.0f;
        int i=0;
        
        for(int readInd = writeInd; readInd >= 0; readInd--)
            output += IR[i++] * buffer[readInd];
        
        for(int readInd = IR.length-1; readInd > writeInd; readInd--)
            output += IR[i++] * buffer[readInd];
        
        writeInd = (writeInd+1) % buffer.length;
        in.setValue((short)output);
    }

}
