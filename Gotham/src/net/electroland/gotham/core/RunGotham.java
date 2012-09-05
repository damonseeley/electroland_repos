package net.electroland.gotham.core;

import javax.swing.JFrame;

import net.electroland.gotham.core.ui.GothamFrame;

public class RunGotham {

    public static void main(String[] args) {
        GothamFrame frame = new GothamFrame();
        frame.setSize(1200, 700);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }
}