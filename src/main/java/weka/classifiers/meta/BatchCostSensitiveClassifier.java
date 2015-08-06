package weka.classifiers.meta;

import weka.core.BatchPredictor;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 * This is a modification of CostSensitiveClassifier that is able to
 * process instances in a batch fashion. Intended for use with
 * PyScriptClassifier since it works slowly when a Python script
 * has to process instances on an instance-by-instance basis.
 * @author cjb60
 *
 */
public class BatchCostSensitiveClassifier extends CostSensitiveClassifier
	implements BatchPredictor {
	
	public BatchCostSensitiveClassifier() {
		super();
		System.out.println("batch classifier");
	}

	private static final long serialVersionUID = -5206049616757248411L;

	@Override
	public void setBatchSize(String size) {
	}

	@Override
	public String getBatchSize() {
		return null;
	}
	
	@Override
	public double[][] distributionsForInstances(Instances insts)
			throws Exception {
		double[][] dists = ((BatchPredictor) m_Classifier).distributionsForInstances(insts);
		// if we are minimizing expected cost, we have to do a bit extra
		if (m_MinimizeExpectedCost) {
			for(int x = 0; x < insts.numInstances(); x++) {
				double[] pred = dists[x];
				double[] costs = m_CostMatrix.expectedCosts(pred, insts.get(x));
				// This is probably not ideal
				int classIndex = Utils.minIndex(costs);
				for (int i = 0; i  < pred.length; i++) {
					if (i == classIndex) {
						pred[i] = 1.0;
					} else {
						pred[i] = 0.0;
					}
				}
				dists[x] = pred;
			}
		
		}
		return dists;
	}
	
	/*
	@Override
	public double[] distributionForInstance(Instance inst) 
			throws Exception {
		Instances insts = new Instances(inst.dataset());	
		return distributionsForInstances(insts)[0];
	}
	*/
	
	public static void main(String[] argv) {
		runClassifier(new BatchCostSensitiveClassifier(), argv);
	}
}
