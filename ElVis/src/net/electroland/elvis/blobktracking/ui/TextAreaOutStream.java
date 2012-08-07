package net.electroland.elvis.blobktracking.ui;

/*
*
* @(#) TextAreaOutputStream.java
*
*/

import java.awt.Color;
import java.io.IOException;
import java.io.OutputStream;

import javax.swing.JTextArea;

public class TextAreaOutStream extends OutputStream {
	Color color;
    private JTextArea textArea;
    
    public TextAreaOutStream( JTextArea textArea, Color c) {
        this.textArea = textArea;
        color = c;
    }
    
    public void write( int b ) throws IOException {
    	textArea.setForeground(color);
        textArea.append( String.valueOf( ( char )b ) );
    }   
 
    
    
}