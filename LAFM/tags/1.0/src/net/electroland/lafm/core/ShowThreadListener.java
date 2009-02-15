package net.electroland.lafm.core;

import java.util.Collection;

import net.electroland.detector.DMXLightingFixture;

public interface ShowThreadListener {
	abstract void notifyComplete(ShowThread showthread, Collection <DMXLightingFixture> flowers);
}