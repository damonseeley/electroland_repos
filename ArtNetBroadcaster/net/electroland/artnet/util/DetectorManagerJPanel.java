package net.electroland.artnet.util;

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

import net.electroland.detector.DMXLightingFixture;
import net.electroland.detector.Detector;
import net.electroland.detector.DetectorManager;

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
		Iterator <DMXLightingFixture> i = dm.getFixtures().iterator();

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
		dm.getFixture((String)fixtureList.getSelectedItem()).setLog(isLogging);
	}
	
	private void newRaster()
	{
		DMXLightingFixture fixture = dm.getFixture((String)fixtureList.getSelectedItem());
		this.raster = new BufferedImage(fixture.getWidth(), fixture.getHeight(), BufferedImage.TYPE_INT_ARGB);		
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
		DMXLightingFixture fixture 
			= dm.getFixture((String)fixtureList.getSelectedItem());
		g.setColor(Color.DARK_GRAY);
		g.drawRect(0, 0, fixture.getWidth(), fixture.getHeight());

		// draw the detectors
		Iterator <Detector> i = fixture.getDetectors().iterator();
		while (i.hasNext())
		{
			Detector detector = i.next();
			g.setColor(Color.WHITE);
			g.drawRect(detector.getX(), detector.getY(), 
						detector.getWidth(), detector.getHeight());
		}

		// do the shit!
		fixture.sync(raster);
	}
}