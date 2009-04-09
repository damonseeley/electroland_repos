package net.electroland.noho.ui;

import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;

import javax.swing.JFrame;


public class TestUI extends Component {
	
	public BufferedImage img;
	
	final static Color bg = Color.black;
	
	public TestUI(BufferedImage img) {
		this.img = img;
		
		//temp, display the image
        JFrame f = new JFrame("Test UI");
        
		f.setUndecorated(true);
        f.setVisible(true);
        
		// remove window frame
		f.setVisible(false);
		
		f.setPreferredSize(new Dimension(390,14));
        
        f.addWindowListener(new WindowAdapter(){
                public void windowClosing(WindowEvent e) {
                    System.exit(0);
                }
            });
        
		f.addMouseListener(new MouseListener() {
			public void mouseClicked(MouseEvent e) {
				System.exit(0);
			}

			public void mousePressed(MouseEvent e) {
			}

			public void mouseReleased(MouseEvent e) {
			}

			public void mouseEntered(MouseEvent e) {
			}

			public void mouseExited(MouseEvent e) {
				//System.exit(0);
			}
		});

        f.add(this);
        f.pack();
      
		f.dispose();
		//f.setUndecorated(true);
        f.setVisible(true);
        
		//switching to fullscreen mode
		//GraphicsEnvironment.getLocalGraphicsEnvironment().getDefaultScreenDevice().setFullScreenWindow(f);

        
	}

    
	// temp jpanel stuff to display image
	public void paint(Graphics g) {
		
		g.setColor(bg);
		g.fillRect(0, 0, 390, 14);
		g.drawImage(img, 0, 0, null);
        
    }
    
    public Dimension getPreferredSize() {
    	// just dimensions the jpanel
    	return new Dimension(390,14);
    }

}
