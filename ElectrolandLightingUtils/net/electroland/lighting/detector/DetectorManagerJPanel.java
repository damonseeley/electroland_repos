package net.electroland.lighting.detector;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.util.Iterator;

import javax.swing.JComboBox;
import javax.swing.JPanel;


@SuppressWarnings("serial")
public class DetectorManagerJPanel extends JPanel implements ActionListener {

	private JComboBox fixtureList;
	private DetectorManager dm;
	private BufferedImage raster;

	public DetectorManagerJPanel(DetectorManager dm)
	{
		this.dm = dm;

		/* list of fixtures */
		fixtureList = new JComboBox();
		Iterator <Recipient> i = dm.getRecipients().iterator();

		while (i.hasNext())
		{
			fixtureList.addItem(i.next().getID());
		}
		
		this.setLayout(new BorderLayout());
		this.add(fixtureList, BorderLayout.SOUTH);
		fixtureList.addActionListener(this);

		newRaster();
	}

	public DetectorManager getDetectorManager()
	{
		return dm;
	}
	
	public void setLog(boolean isLogging)
	{
		System.out.println("is logging=" + isLogging);
		dm.getRecipient((String)fixtureList.getSelectedItem()).setLog(isLogging);
	}
	
	private void newRaster()
	{
		Recipient fixture = dm.getRecipient((String)fixtureList.getSelectedItem());
		this.raster = new BufferedImage(fixture.getPreferredDimensions().width, 
										fixture.getPreferredDimensions().height, 
										BufferedImage.TYPE_INT_ARGB);
	}

	public Image getRaster()
	{
		return this.raster;
	}
	
	public void actionPerformed(ActionEvent e)
	{
		newRaster();
		repaint();
    }

	@Override
	protected void paintComponent(Graphics g) 
	{
		super.paintComponent(g);

		g.setColor(Color.BLACK);
		g.fillRect(0, 0, this.getWidth(), this.getHeight());

		// draw raster
		g.drawImage(raster, 0, 0, raster.getWidth(this), raster.getHeight(this), this);
		
		// draw the border of the fixture
		Recipient fixture 
			= dm.getRecipient((String)fixtureList.getSelectedItem());
		g.setColor(Color.DARK_GRAY);
		g.drawRect(0, 0, fixture.getPreferredDimensions().width, fixture.getPreferredDimensions().height);

		// draw the detectors
		Iterator <Detector> i = fixture.getDetectors().iterator();
		while (i.hasNext())
		{
			Detector detector = i.next();
			if (detector != null)
			{
				g.setColor(Color.WHITE);
				g.drawRect(detector.getX(), detector.getY(), 
							detector.getWidth(), detector.getHeight());
			}
		}

		// do the shit!
		fixture.sync(raster);
	}
}