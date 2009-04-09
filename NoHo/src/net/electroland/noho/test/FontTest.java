package net.electroland.noho.test;

import java.awt.Component;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.GraphicsEnvironment;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

import javax.swing.JFrame;

import net.electroland.noho.core.fontMatrix.FontMatrix;


public class FontTest extends Component {

	static FontMatrix fontStandard;
	
	public static void main(String[] args) {
		
		//testFont = new LetterformImage("A_2.gif");
		fontStandard = 
			new FontMatrix("C:/Documents and Settings/Damon/My Documents/_ELECTROLAND/_PROJECTS/NoHo/DEV/TYPEFACE DEV/FINAL MATRIX FONT/5x7 STD GIF/");

		//temp, display the image
        JFrame f = new JFrame("Load Image Sample");
        
        f.addWindowListener(new WindowAdapter(){
                public void windowClosing(WindowEvent e) {
                    System.exit(0);
                }
            });
       

        f.add(new FontTest());
        f.pack();
        f.setVisible(true);

        


	}
	
    
	// temp jpanel stuff to display image
	public void paint(Graphics g) {
        
		//System.out.println("painting now");
		int charWidth = 10;
		
		String aMessage = "you can always spot the equinoxes";
		aMessage = aMessage.toLowerCase();
		
		for (int charNum = 0; charNum<39; charNum++){
			if (charNum < aMessage.length()) {
			char c = aMessage.charAt(charNum);
			//int thecharint = (int)c;
			//System.out.println(c);
			   g.drawImage(fontStandard.getLetterformImg(c), charNum*charWidth, 0, null);
			} else {
				g.drawImage(fontStandard.getLetterformImg(' '), charNum*charWidth, 0, null);
			}

		}
        
    }
    
    public Dimension getPreferredSize() {
    	// just dimensions the jpanel
    	return new Dimension(390,14);
    }

}
