package net.electroland.noho.z_temp;
import java.awt.image.*;
import java.awt.*;
import java.applet.*;
import java.awt.event.*; 


final class MyGifPixelGrabber implements ImageObserver
{
  private Image m_image=null;   // pointer to original image
  private Object m_pixels=null; // either array of ints or bytes
  private int m_iNumOfColors=0;  
  private int m_iWidth, m_iHeight;
  private ColorModel m_colorModel=null;

  public boolean isIndexed()
  {
	if (m_colorModel==null) { return false; }
    return ((m_colorModel instanceof IndexColorModel));
  }
  
  public MyGifPixelGrabber(Image img)
  {
    m_image=img;
  }

  public int getNumOfColors()
  {
    return m_iNumOfColors;
  }

  public void destroy()
  {
    m_image=null;
    m_pixels=null;
  }

  public void grabPixels()
  {
    m_iWidth=m_image.getWidth(this);
    m_iHeight=m_image.getHeight(this); 
    PixelGrabber pixelGrabber=new PixelGrabber(m_image, 0,0, m_iWidth, m_iHeight, false);  

    try
    {
      pixelGrabber.grabPixels();
    }
    catch (Exception e)
    {
      System.out.println("PixelGrabber exception"); 
    }
    m_pixels=(Object)pixelGrabber.getPixels();
    // get the palette of the image
    m_colorModel=pixelGrabber.getColorModel();
    if (!(m_colorModel instanceof IndexColorModel))
    {
      // not an indexed file (ie: not a gif file)
    }
    else
    {
      m_iNumOfColors=((IndexColorModel)m_colorModel).getMapSize();
    }
  }

  // you'd need to cast the return values, which will be an array of bytes
  // or an array of ints.  if the file is a gif file, it will return an 
  // array of bytes, if jpg, you will get an array of ints
  public Object getPixels()
  {
    return m_pixels; 
  }

  public int getWidth()
  {
    return m_iWidth;
  }

  public int getHeight()
  {
    return m_iHeight;
  }
  
  public ColorModel getColorModel()
  {
	  return m_colorModel;
  }
  
  public int getRed(int pixel)
  {
    if ((m_colorModel instanceof IndexColorModel))	  
		return ((IndexColorModel)m_colorModel).getRed(pixel);
	else
		return ((DirectColorModel)m_colorModel).getRed(pixel);
  }

  public int getGreen(int pixel)
  {
    if ((m_colorModel instanceof IndexColorModel))	  
		return ((IndexColorModel)m_colorModel).getGreen(pixel);
	else
		return ((DirectColorModel)m_colorModel).getGreen(pixel);
  }
  
  public int getBlue(int pixel)
  {
    if ((m_colorModel instanceof IndexColorModel))	  
		return ((IndexColorModel)m_colorModel).getBlue(pixel);
	else
		return ((DirectColorModel)m_colorModel).getBlue(pixel);
  }
  
  public int getRGB(int pixel)
  {
    if ((m_colorModel instanceof IndexColorModel))	  
		return ((IndexColorModel)m_colorModel).getRGB(pixel);
	else
		return pixel;
  }
  
  // we need this method just because we're extending ImageObserver.
  public boolean imageUpdate(Image img, int infoflags, int x, int y, int width, int height) 
  {
    return true;	  
  }  
}




class MyFrame extends Frame
{
  MyFrame()
  {
    super("PixelData");
  }

  public boolean handleEvent(Event e)
  {
    if (e.id==Event.WINDOW_DESTROY)
    {
      dispose();
      return false;
    }
    return super.handleEvent(e);
  }
}



