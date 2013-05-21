package net.electroland.gotham.processing.metaballs;

import java.awt.Color;
import java.awt.Rectangle;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import net.electroland.ea.EasingFunction;
import net.electroland.elvis.net.GridData;
import net.electroland.gotham.processing.GothamPApplet;
import net.electroland.gotham.processing.assets.FastBlur;
import net.electroland.utils.ElectrolandProperties;

import org.apache.log4j.Logger;

import processing.core.PVector;

@SuppressWarnings("serial")
public class Metaballs extends GothamPApplet {

    static Logger logger = Logger.getLogger(Metaballs.class);

    private MetaballsProps props;
    private List <MetaballGroup>groups;
    private GridData gridData;
    private GridHistory gridHistory;
    private WindController windController;
    private List <Jet>jets;

    @Override
    public void setup(){

        String name = this.getProperties().getRequired("name");
        ElectrolandProperties globalProps = new ElectrolandProperties("Gotham-global.properties");
        props = new MetaballsProps(this, name, globalProps);

        gridHistory = new GridHistory((int)props.getValue(MetaballsProps.GRID_HISTORY));
        groups = new ArrayList<MetaballGroup>();
        jets = new ArrayList<Jet>();
        windController = new WindController(globalProps.getDefaultDouble(name, "wind", "minSecondsBetweenGusts", 0).floatValue(),
                globalProps.getDefaultDouble(name, "wind", "maxSecondsBetweenGusts", 0).floatValue());

        // load balls, jets, and wind
        synchronized(groups){
            for (String objectName : globalProps.getObjectNames(name)){

                if (objectName.startsWith("balls.")){

                    int id              = Integer.parseInt(objectName.substring(6, objectName.length()));
                    float hueVariance   = props.getValue(MetaballsProps.HUE_VARIANCE);
                    MetaballGroup group = new MetaballGroup(id, hueVariance);
                    groups.add(group);
                    for (String size : globalProps.getRequiredList(name, objectName, "sizes")){
                        Metaball ball = new Metaball(Integer.parseInt(size));
                        ball.setCenter(new PVector(random(0, this.getSyncArea().width), 
                                                   random (0, this.getSyncArea().height)));
                        ball.setVelocity(new PVector(random(-1,1), random(-1,1)));
                        group.add(ball);
                        logger.info("loaded: " + ball);
                    }

                } else if (objectName.startsWith("jet.")){

                    float x             = globalProps.getRequiredDouble(name, objectName, "x").floatValue();
                    float y             = globalProps.getRequiredDouble(name, objectName, "y").floatValue();
                    float angle         = globalProps.getRequiredDouble(name, objectName, "degreesFromNorthClockwise").floatValue();
                    float baseForce     = globalProps.getRequiredDouble(name, objectName, "baseForce").floatValue();
                    float seconds       = globalProps.getRequiredDouble(name, objectName, "maxDurationSeconds").floatValue();
                    boolean allowReverse= globalProps.getDefaultBoolean(name, objectName, "allowReverse", false);
                    Jet jet             = new Jet(new PVector(x, y), angle, baseForce, seconds, allowReverse);
                    logger.info("loaded: " + jet);
                    jets.add(jet);

                } else if (objectName.startsWith("wind.")){

                    float degrees       = globalProps.getRequiredDouble(name, objectName, "degreesFromNorthClockwise").floatValue();
                    float degreesFinal  = globalProps.getRequiredDouble(name, objectName, "finalDegrees").floatValue();
                    float max           = globalProps.getRequiredDouble(name, objectName, "maxStrength").floatValue();
                    float holdSecs      = globalProps.getRequiredDouble(name, objectName, "holdSeconds").floatValue();
                    float entranceSecs  = globalProps.getRequiredDouble(name, objectName, "entranceSeconds").floatValue();
                    EasingFunction in   = (EasingFunction)globalProps.getRequiredClass(name, objectName, "entranceEasingFunction");
                    float exitSecs      = globalProps.getRequiredDouble(name, objectName, "exitSeconds").floatValue();
                    EasingFunction out  = (EasingFunction)globalProps.getRequiredClass(name, objectName, "exitEasingFunction");
                    Wind w = new Wind(degrees, degreesFinal,  max, holdSecs, entranceSecs, in, exitSecs, out);
                    logger.info("loaded: " + w);
                    windController.addWind(w);

                }
            }
        }
    }

    @Override
    public void drawELUContent() {

        this.syncLocalVariablesWithPropsControls();
        boolean gridIsAffectingBalls  = props.getState(MetaballsProps.ENABLE_GRID);
        boolean showGrid              = props.getState(MetaballsProps.SHOW_GRID);
        boolean showWind              = props.getState(MetaballsProps.SHOW_WIND);

        Wind wind                     = windController.next();

        // TODO: need to age gridData even if camera stops, so things don't get stuck.  in otherword, if
        //  a NEW gridData didn't come in, pass an empty gridData to occuped blocks every frame.
        this.gridHistory.addData(getOccupiedBlocks(gridData, props.getBoundary(MetaballsProps.GRID_ON_CANVAS)));

        this.renderBackground(props.getColor(MetaballsProps.BG_COLOR));

        for (MetaballGroup group : groups){

            group.advanceIndividualBalls();
            group.applyFriction(props.getValue(MetaballsProps.FRICTION, group.id));
            group.applyPointForce(getCanvasCenter(), props.getValue(MetaballsProps.CENTER_FORCE, group.id));
            group.applyGroupCohesion(props.getValue(MetaballsProps.COHESIVENESS, group.id));
            group.applyBallToBallRepellForces(groups, props.getValue(MetaballsProps.BALL_TO_BALL_REPELL, group.id));
            //group.applyGroupToGroupRepellForces(groups, props.getValue(MetaballsProps.BALL_TO_BALL_REPELL));
            group.applyJetForces(jets, props.getValue(MetaballsProps.JET_FORCE_SCALE, group.id));
            if (wind != null){
                group.applyWindForces(wind, props.getValue(MetaballsProps.WIND_FORCE_SCALE, group.id));
            }

            if (gridIsAffectingBalls){
                group.applyPresenceImpactForces(gridHistory.latest(), props.getValue(MetaballsProps.REPELL_FORCE, group.id));
            }
            group.constrainVelocity(props.getValue(MetaballsProps.MIN_VELOCITY, group.id),  // first max/min check is so wind, other balls, etc. don't cause
                    props.getValue(MetaballsProps.MAX_VELOCITY, group.id)); // crazy energy

            group.keepBallsWithinBoundary(this.getBoundary());

            this.renderBallGroup(group);
        }

        if (gridIsAffectingBalls) {
            if (showGrid){
                this.renderGrid(props.getBoundary(MetaballsProps.GRID_ON_CANVAS));
            } else {
                this.renderGhostlyPresence((int)props.getValue(MetaballsProps.PRESENCE_OPACITY), 
                                            props.getValue(MetaballsProps.PRESENCE_RADIUS));
            }
        }

        if (!showGrid){
            this.blurRendering();
        }

        noFill();
        stroke(Color.WHITE.getRGB());
        Rectangle boundary = getBoundary();
        rect(boundary.x, boundary.y, boundary.width, boundary.height);
                
        
        if (wind != null && showWind){
            renderWind(wind);
        }

        renderJets(jets);
    }

    private Rectangle getBoundary(){
        int margin = (int)props.getValue(MetaballsProps.MARGIN);
        return new Rectangle(margin, margin, 
                             this.getSyncArea().width - (2 * margin),
                             this.getSyncArea().height - (2 * margin));
    }

    private PVector getCanvasCenter(){
        return new PVector(this.getSyncArea().width / 2, this.getSyncArea().height / 2);
    }

    private void renderJets(List<Jet> jets){
        this.noFill();
        this.stroke(Color.WHITE.getRGB());
        for (Jet j : jets){
            line(j.getOrigin().x, j.getOrigin().y - 5, j.getOrigin().x, j.getOrigin().y + 5);
            line(j.getOrigin().x - 5, j.getOrigin().y, j.getOrigin().x + 5, j.getOrigin().y);
        }
    }

    private void renderWind(Wind wind){
        PVector angle = wind.getSource();
        PVector widgetLocation = new PVector (this.getSyncArea().width + 5, 
                                                this.getSyncArea().height + 5);
        angle.mult(wind.getStrength());
        angle.mult(100);
        angle.add(widgetLocation);
        this.noFill();
        this.stroke(Color.WHITE.getRGB());
        this.translate(angle, widgetLocation);
        this.line(angle.x, angle.y, widgetLocation.x, widgetLocation.y);
    }

    private void translate(PVector moving, PVector around){
        PVector diff = PVector.sub(around, moving);
        moving.add(diff);
        moving.add(diff);
    }

    private void renderBackground(Color bgColor){
        fill(bgColor.getRed(), bgColor.getGreen(), bgColor.getBlue(), 255);
        rect(0, 0, width, height);
    }

    private void renderBallGroup(MetaballGroup group){
        for (Metaball ball : group.balls){
            Color ballColor = ball.getColor();
            this.fill(ballColor.getRed(), ballColor.getGreen(), 
                        ballColor.getBlue(), group.getOpacity());
            this.noStroke();
            this.ellipse(ball.getCenter().x, ball.getCenter().y,
                         ball.width(),
                         ball.height());
        }
    }

    private void renderGrid(Rectangle gridCanvas){
        if (gridData != null){

            float cellWidth  = gridCanvas.width / (float)gridData.width;
            float cellHeight = gridCanvas.height / (float)gridData.height;

            stroke(255); // gridline

            for (int x = 0; x < gridData.width; x++){
                for (int y = 0; y < gridData.height; y++){

                    if (gridData.getValue(x, y) != (byte)0){
                        fill(255); // light up occupied point
                    }else{
                        noFill();  // don't light up non-occupied point
                    }

                    this.rect(gridCanvas.x + (x * cellWidth), 
                                gridCanvas.y + (y * cellHeight), 
                                cellWidth, cellHeight);
                }
            }
        }
    }

    private void renderGhostlyPresence(int opacity, float radius){

        noStroke();
        fill(255, 255, 255, opacity);

        for (List<PVector> points : gridHistory){
            for (PVector point : points){
                this.rect(point.x, point.y, radius, radius);
            }
        }
    }

    private void blurRendering(){

        loadPixels();
        FastBlur.performBlur(pixels, width, height, floor(props.getValue(MetaballsProps.BLUR)));
        updatePixels();
    }

    public void syncLocalVariablesWithPropsControls(){
        for (MetaballGroup group : groups){
            group.setBaseColor(props.getColor(group.id));
            group.setOpacity((int)props.getValue(MetaballsProps.BALL_OPACITY));
            group.setScale(props.getValue(MetaballsProps.BALL_SCALE));
        }
        this.setGoToBlack(props.getState(MetaballsProps.GO_TO_BLACK));
        gridHistory.setMaxLength((int)props.getValue(MetaballsProps.GRID_HISTORY));
    }

    public List<PVector> getOccupiedBlocks(GridData grid, Rectangle gridCanvas){
        if (grid == null){
            return Collections.<PVector>emptyList();
        }

        float cellWidth  = gridCanvas.width / (float)gridData.width;
        float cellHeight = gridCanvas.height / (float)gridData.height;

        List<PVector>objects = new ArrayList<PVector>();
        for (int i = 0; i < grid.data.length; i++){
            if (grid.data[i] != (byte)0){
                int y = i / grid.width;
                int x = i - (y * grid.width);
                objects.add(new PVector(cellWidth * x + gridCanvas.x,
                                        cellHeight * y + gridCanvas.y));
            }
        }
        return objects;
    }

    @Override
    public void handle(GridData srcData) {
        
        if (gridData == null){
            gridData = srcData;
        } else {
            // copy the original source, so we don't accidentally change
            // the source for other clients using this data.
            StringBuilder sb = new StringBuilder();
            srcData.buildString(sb);
            srcData = new GridData(sb.toString());

            if (props != null){
                srcData = GridData.subset(srcData, props.getBoundary(MetaballsProps.GRID));

                srcData = GridData.counterClockwise(srcData);

                if (props.getState(MetaballsProps.MIRROR_HORIZONTAL)){
                    srcData = GridData.flipHorizontal(srcData);
                }
                if (props.getState(MetaballsProps.MIRROR_VERTICAL)){
                    srcData = GridData.flipVertical(srcData);
                }
                gridData = srcData;
            }
        }
    }

    public void dumpStateToConsole(){
        for (MetaballGroup group : groups){
            for (Metaball ball : group.balls){
                System.out.println(ball);
            }
        }
    } 
}