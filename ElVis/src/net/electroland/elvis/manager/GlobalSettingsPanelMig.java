package net.electroland.elvis.manager;

import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.io.File;
import java.io.IOException;
import java.util.Hashtable;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JFormattedTextField;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.JTextField;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.text.NumberFormatter;

import net.electroland.elvis.regions.GlobalRegionSnapshot;
import net.miginfocom.swing.MigLayout;

public class GlobalSettingsPanelMig implements Colorable {

	JComboBox urlBox;
	String urlSelected;

	JComboBox imageView;
	String imageViewSelected = "";


	JSlider triggerThreshSlider;
	JTextField triggerTreshField;


	JSlider backgroundAdaptSlider;
	JTextField backgroundAdaptField;



	ColorButton colorButton;


	public static GlobalSettingsPanelMig thePanel;

	public GlobalSettingsPanelMig() {
		thePanel = this;


	}

	public JPanel build() {
		JPanel p = new JPanel(new MigLayout("", "[left]"));

		p.add(new JLabel("URL/SRC:"), "align right");


		urlBox = new JComboBox();
		urlBox.addItem("None");
		urlBox.addItem("Image");
		urlBox.addItem(ImagePanel.NOHONORTH_SRC);
		urlBox.addItem(ImagePanel.NOHOSOUTH_SRC);
		urlBox.addItem(ImagePanel.FLOWER_SRC);
		urlBox.addItem(ImagePanel.LOCALAXIS_SRC);
		urlBox.addItem(ImagePanel.JMYRON_SRC);
		urlBox.setSelectedItem("None");
		urlBox.setMaximumRowCount(40);
		urlBox.addItemListener(new ItemListener() {
			public void itemStateChanged(ItemEvent e) {
				String selected = (String) urlBox.getSelectedItem();
				if(selected.equals(urlSelected)) return; // no change
				urlSelected = selected;
//				System.out.println(selected);
				if(urlSelected.equals("Image")) {
	//				System.out.println("its an image");
					JFileChooser fc = new JFileChooser ();
					fc.setDialogTitle ("Open Image");
					fc.setFileSelectionMode ( JFileChooser.FILES_ONLY);
					fc.showOpenDialog(null);
					File f = fc.getSelectedFile();
					if(f == null) {
						urlBox.setSelectedItem("None");
					} else {
//						System.out.println("Opening " + f.getAbsolutePath());
						urlBox.setSelectedItem(f.getAbsolutePath().toString());
						try {
							ImagePanel.THE_IMAGEPANEL.setBackgroundImage(f);
						} catch (IOException e1) {
							System.out.println(e1);
							urlBox.setSelectedItem("None");							
						}
					}

				} else {
					try {
						ImagePanel.THE_IMAGEPANEL.setBackgroundStream(urlSelected);
					} catch (IOException e1) {
						urlBox.setSelectedItem("None");							
					}
				}
			}});

		p.add(urlBox, "wrap");

		p.add(new JLabel("View:"), "align right");

		imageView =new JComboBox();
		imageView.setName("view");
		imageView.addItem(ImagePanel.RAW_IMG);
		imageView.addItem(ImagePanel.GRAYSCALE_IMG);
		imageView.addItem(ImagePanel.BACKGROUND_IMG);
		imageView.addItem(ImagePanel.BACKDIFF_IMG);
		imageView.addItem(ImagePanel.THRESHOLD_IMG);
		imageView.addItemListener(new ItemListener() {
			public void itemStateChanged(ItemEvent e) {
				String newSelect = (String)imageView.getSelectedItem();
				if(! imageViewSelected.equals(newSelect)) {
					imageViewSelected = newSelect;
					ImagePanel.THE_IMAGEPANEL.setImageViewType(imageViewSelected);
				}

			}

		});

		p.add(imageView, "wrap");




		p.add(new JLabel("Diff thresh:"),  "gaptop 25, align right");


		java.text.NumberFormat numberFormat =
			java.text.NumberFormat.getIntegerInstance();
		NumberFormatter formatter = new NumberFormatter(numberFormat);
		formatter.setMinimum(new Integer(0));
		formatter.setMaximum(new Integer(65535));
		triggerTreshField = new JFormattedTextField(formatter);
		triggerTreshField.setText("30000");
		triggerTreshField.setColumns(10); //get some space

		p.add(triggerTreshField, "gaptop 25, align left, wrap");

		triggerThreshSlider = new JSlider(0,65535,30000);

		Hashtable<Integer, JLabel> h = new Hashtable<Integer, JLabel>();
		h.put(new Integer(0),new JLabel("0"));
		h.put(new Integer(65535),new JLabel("65535"));




		triggerThreshSlider.setLabelTable(h);
		triggerThreshSlider.setPaintLabels(true);


		triggerTreshField.addActionListener(new ActionListener() {

			public void actionPerformed(ActionEvent e) {
				int i = Integer.parseInt(triggerTreshField.getText());
				triggerThreshSlider.setValue(i);
				ImagePanel.THE_IMAGEPANEL.setThresh(i);

			}

		});
		triggerThreshSlider.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				int i = triggerThreshSlider.getValue();
				triggerTreshField.setText(Integer.toString(i));
				ImagePanel.THE_IMAGEPANEL.setThresh(i);
			}

		});
		p.add(triggerThreshSlider, "span, wrap");


		p.add(new JLabel("Adaptation:"),  "gaptop 25, align right");


		numberFormat =java.text.NumberFormat.getNumberInstance();
		formatter = new NumberFormatter(numberFormat);
		formatter.setMinimum(new Double(0));
		formatter.setMaximum(new Double(1.0));

		backgroundAdaptField = new JFormattedTextField(formatter);
		backgroundAdaptField.setText(".01");
		backgroundAdaptField.setColumns(10); //get some space

		p.add(backgroundAdaptField, "gaptop 25, align left, wrap");

		backgroundAdaptSlider = new JSlider(0, 10000, 100);

		h = new Hashtable<Integer, JLabel>();
		h.put(new Integer(0),new JLabel("0.0"));
		h.put(new Integer(10000),new JLabel("1.0"));



		backgroundAdaptSlider.setLabelTable(h);
		backgroundAdaptSlider.setPaintLabels(true);


		backgroundAdaptField.addActionListener(new ActionListener() {

			public void actionPerformed(ActionEvent e) {
				double d = Double.parseDouble(backgroundAdaptField.getText());
				backgroundAdaptSlider.setValue((int) (d * 10000.0));
				ImagePanel.THE_IMAGEPANEL.setAdaptation(d);
			}

		});
		backgroundAdaptSlider.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				double d = ((double) backgroundAdaptSlider.getValue()) / 10000.0;
				backgroundAdaptField.setText( Double.toString(d));
				System.out.println("setting adaption to" + d);
				ImagePanel.THE_IMAGEPANEL.setAdaptation(d);
			}

		});
		p.add(backgroundAdaptSlider, "span, wrap");




		p.add(new JLabel("Region Color:"), "gaptop 25, align right");
		colorButton = new ColorButton(65,40,this);
		colorButton.setColor(Color.RED);
		p.add(colorButton, "gaptop 25, align left, wrap");


		JButton b = new JButton("Open");
		b.addActionListener(new ActionListener() {

			public void actionPerformed(ActionEvent e) {
				GlobalRegionSnapshot gsr = GlobalRegionSnapshot.load();
				ImagePanel.THE_IMAGEPANEL.setAdaptation(gsr.backgroundAdaptation);
//				System.out.println("adapt " + gsr.backgroundAdaptation);
				backgroundAdaptSlider.setValue((int) (gsr.backgroundAdaptation * 10000.0));
				backgroundAdaptField.setText(Double.toString(gsr.backgroundAdaptation));
				
				ImagePanel.THE_IMAGEPANEL.setThresh(gsr.backgroundDiffThresh);
				triggerThreshSlider.setValue((int)gsr.backgroundDiffThresh);
				triggerTreshField.setText(Integer.toString((int)gsr.backgroundDiffThresh));
				ImagePanel.THE_IMAGEPANEL.regions = gsr.regions;

			}});
		p.add(b, "gaptop 15, align left");
		
		b = new JButton("Save");
		b.addActionListener(new ActionListener() {

			public void actionPerformed(ActionEvent e) {
				GlobalRegionSnapshot gsr = new GlobalRegionSnapshot(ImagePanel.THE_IMAGEPANEL.w, ImagePanel.THE_IMAGEPANEL.h, ImagePanel.THE_IMAGEPANEL.getAdaptation(), ImagePanel.THE_IMAGEPANEL.getThresh(), ImagePanel.THE_IMAGEPANEL.regions);
				gsr.save();
			}});
		p.add(b, "gaptop 15, align right");
		return p;

	}

	public Color getColor() {
		if(ImagePanel.THE_IMAGEPANEL.selectedRegion != null) {
			return ImagePanel.THE_IMAGEPANEL.getColor();
		} else {
			return null;
		}
	}

	public void setColor(Color c) {
		if(ImagePanel.THE_IMAGEPANEL.selectedRegion != null) {
			ImagePanel.THE_IMAGEPANEL.setColor(c);
		}		
		ImagePanel.THE_IMAGEPANEL.repaint();


	}
	


}

