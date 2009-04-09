package net.electroland.noho.z_temp;
import java.applet.Applet;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.MediaTracker;
import java.awt.TextArea;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.image.DirectColorModel;

public class TestImagePixels extends Applet implements Runnable, MouseListener
{
  Image m_image=null;
  Object m_pixelData=null;
  private String m_imageFilename;
  private Thread m_thread=null;
  private String m_message=null;
  private MyGifPixelGrabber m_gifPixelGrabber=null;
  
  public void start()
  {
	  m_thread=new Thread(this);
	  m_thread.start();
  }
  
  public void stop()
  {
  }

  public void init()
  {
	m_imageFilename=getParameter("A.gif");
    // load image and make sure it's loaded before continuing.
    // (this is not a good way to do image loading, it's just for
    // illustration purpose.  on a real application, you might 
    // create a separate thread to load the image so that it won't
    // tie up the applet)
    m_image=getImage(getCodeBase(), m_imageFilename);
    MediaTracker mediaTracker=new MediaTracker(this);
    mediaTracker.addImage(m_image, 0);
    try
    {
      mediaTracker.waitForID(0);	
    }
    catch (InterruptedException e)
    {
    }
    mediaTracker.removeImage(m_image);
    m_gifPixelGrabber=new MyGifPixelGrabber (m_image);
    m_gifPixelGrabber.grabPixels();
	m_pixelData=m_gifPixelGrabber.getPixels();
	MyFrame f=new MyFrame();
	f.resize(200,300);
	TextArea textArea=new TextArea();
	textArea.setFont(new Font("Courier", 12,12));
    textArea.append(m_imageFilename+"\n");
	textArea.append("Width="+m_gifPixelGrabber.getWidth()+"\n");
	textArea.append("Height="+m_gifPixelGrabber.getHeight()+"\n");	
	if (m_gifPixelGrabber.isIndexed())	
		textArea.append("NumOfColors="+m_gifPixelGrabber.getNumOfColors()+"\n");		
	f.add("Center", textArea);
	f.show();
	if (m_gifPixelGrabber.isIndexed())	
	{
		byte[] pixelData=(byte[])m_pixelData;
		for (int i=0; m_gifPixelGrabber.isIndexed() && i<m_gifPixelGrabber.getHeight(); i++)
		{
			String s="";
			for (int j=0; j<m_gifPixelGrabber.getWidth(); j++)
		    {
				if ((int)pixelData[i*m_gifPixelGrabber.getWidth()+j]!=0)
				  s+=(int)pixelData[i*m_gifPixelGrabber.getWidth()+j];
				else
				  s+=" ";
			}
			textArea.appendText(s+"\n");
		}
	}
  }

  public void run()
  {
	addMouseListener(this);
	while (true)
	{
		try
		{
		  repaint();
		  Thread.sleep(100);
		}
		catch (InterruptedException e)
		{
		}		
	}
  }
  
  public void update(Graphics g)
  {
	g.setColor(Color.gray);
	g.fillRect(0,0, size().width, size().height);
    if (m_image!=null)
      g.drawImage(m_image,0,0, this);
	g.setColor(Color.black);
	g.drawString(m_message, 10,120);
	
  }
  
  public void mouseClicked(MouseEvent evt)
  {
	if (evt.getX()< m_image.getWidth(this) && evt.getY()<m_image.getHeight(this) )
	{
		if (m_gifPixelGrabber.isIndexed())
		{
			byte[] pixelData=(byte[])m_pixelData;			
			int pixel=(int)pixelData[evt.getY()*m_image.getWidth(this)+evt.getX()];
			m_message="idx="+pixel+" R="+m_gifPixelGrabber.getRed(pixel)+" G="+m_gifPixelGrabber.getGreen(pixel)+" B="+m_gifPixelGrabber.getBlue(pixel);
			
		}
		else
		{
			DirectColorModel dcm=(DirectColorModel)m_gifPixelGrabber.getColorModel();
			int[] pixelData=(int[])m_pixelData;						
			int pixel=(int)pixelData[evt.getY()*m_image.getWidth(this)+evt.getX()];
			m_message="pixel R="+dcm.getRed(pixel)+" G="+dcm.getGreen(pixel)+" B="+dcm.getBlue(pixel);
		}
	}
	update(getGraphics());
  }
  
  public void mouseEntered(MouseEvent evt){};
  public void mouseExited(MouseEvent evt){};
  public void mousePressed(MouseEvent evt){};  
  public void mouseReleased(MouseEvent evt){};
  
  public static void main (String[] args) {
	  
	  TestImagePixels tip = new TestImagePixels();
	  
	  
  }
}




