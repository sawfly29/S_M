package org.deeplearning4j.examples.convolution;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Created by MakhrovSS on 24.03.2017.
 */
public class ObjectsClassification {
    protected static final Logger log = LoggerFactory.getLogger(ObjectsClassification.class);
    // высотка изображения
    protected static int height = 192;
    // ширина изображения
    protected static int width = 128;
    // количество каналов = 3, так как R G B
    protected static int channels = 3;
    // количество классов классификации
    protected static int numLabels = 4;
    // размер батча
    protected static int batchSize = 20;

    // инициализатор генератора случайных чисел
    protected static long seed = 42;
    // задаем генератор случайных чисел
    protected static Random rng = new Random(seed);
    // частота опроса состояния сети (как часто в консоли будет отражаться ход обучения или тестирования)
    protected static int listenerFreq = 1;
    // количество итераций (прочитайте что такое итерация в контексте данного фреймворка https://deeplearning4j.org/core-concepts#training-a-model )
    protected static int iterations = 1;
    // количество эпох
    protected static int epochs = 50;
    // узазываем, какой процент из образов отвести на обучающую выборку. В данном случае - 80%
    protected static double splitTrainTest = 0.8;
    // количество ядер процессора
    protected static int nCores = 8;
    // сохранять ли состояние синаптических весов после обучения?
    protected static boolean save = true;

    public void run(String[] args) throws Exception {

        log.info("Load data....");
        /**cd
         * Data Setup -> organize and limit data file paths:
         *  - mainPath = path to image files
         *  - fileSplit = define basic dataset split with limits on format
         *  - pathFilter = define additional file load filter to limit size and balance batch content
         **/
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        // указываем путь к образам
        File mainPath = new File(System.getProperty("user.dir"), "dl4j-examples/src/main/resources/faces/");
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        RandomPathFilter pathFilter = new RandomPathFilter(rng, NativeImageLoader.ALLOWED_FORMATS, 0);

        /**
         * Data Setup -> train test split
         *  - inputSplit = define train and test split
         **/
        // создаем две выборки - обучающая выборка - 80% образов, а тестовая - 20% образов из всех в папке с образами
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1.0 - splitTrainTest);
        InputSplit trainData = inputSplit[0]; // обучающая выборка
        InputSplit testData = inputSplit[1]; // тестовая выборка

        /**
         * Data Setup -> transformation
         *  - Transform = how to tranform images and generate large dataset to train on
         **/
        // Определяем дополнительные трансформации изображений (отражения/ искажения), на которых также будет обучать сеть
        ImageTransform flipTransform1 = new FlipImageTransform(rng);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
        ImageTransform warpTransform = new WarpImageTransform(rng, 42);
//        ImageTransform colorTransform = new ColorConversionTransform(new Random(seed), COLOR_BGR2YCrCb);
        // Создаем список трансформаций
        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[]{flipTransform1, warpTransform, flipTransform2});

        /**
         * Data Setup -> normalization
         *  - how to normalize images and generate large dataset to train on
         **/
        // Определяем объект, который нормализует/масштабирует изображения в один размер
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        log.info("Build model....");

        // Присваиваем конфигурацию сети
        MultiLayerNetwork network = alexnetModel();
        network.init();
        network.setListeners(new ScoreIterationListener(listenerFreq));

        /**
         * Data Setup -> define how to load data into net:
         *  - recordReader = the reader that loads and converts image data pass in inputSplit to initialize
         *  - dataIter = a generator that only loads one batch at a time into memory to save memory
         *  - trainIter = uses MultipleEpochsIterator to ensure model runs through the data for all epochs
         **/
        // Опредялем объект, который будет читать изображения
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        DataSetIterator dataIter;
        MultipleEpochsIterator trainIter;


        log.info("Train model....");
        long timeX = System.currentTimeMillis();
        // Train without transformations
        // Инициализируем объект для чтения изображений обучающей выборкой
        recordReader.initialize(trainData, null);
        // Инициализируем итератор по обучающей выборке с указанием размера батча и количества классов
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        // масштабируем изображения
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        // Инициализируем итератор по обучающей выборке с указанием количества эпох, итератора dataIter
        trainIter = new MultipleEpochsIterator(epochs, dataIter, nCores);
        // Обучаем сеть
        network.fit(trainIter);

        // Обучаем сеть на трансформациях
        for (ImageTransform transform : transforms) {
            System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
            recordReader.initialize(trainData, transform);
            dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
            scaler.fit(dataIter);
            dataIter.setPreProcessor(scaler);
            trainIter = new MultipleEpochsIterator(epochs, dataIter, nCores);
            network.fit(trainIter);
        }

        long timeY = System.currentTimeMillis();

        log.info("*** Training complete, time: {} ***", (timeY - timeX));

        // Проверяем работу сети на обучающей выборке
        log.info("Evaluate model....");
        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        Evaluation eval = network.evaluate(dataIter);
        log.info(eval.stats(true));

        // Example on how to get predict results with trained model
        dataIter.reset();
        DataSet testDataSet = dataIter.next();
        String expectedResult = testDataSet.getLabelName(0);
        List<String> predict = network.predict(testDataSet);
        String modelResult = predict.get(0);
        System.out.print("\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelResult + "\n\n");

        if (save) {
            log.info("Save model....");
            String basePath = FilenameUtils.concat(System.getProperty("user.dir"), "dl4j-examples/src/main/resources/");
            ModelSerializer.writeModel(network, basePath + "model.bin", true);
        }
        log.info("****************Example finished********************");
    }

    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

    private SubsamplingLayer maxPool(String name, int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
    }

    private DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build();
    }

    public MultiLayerNetwork alexnetModel() {
        /**
         * AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
         * and the imagenetExample code referenced.
         * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
         **/

        double nonZeroBias = 1;//смещение
        double dropOut = 0.5;//метод улучшения нс для отключения некоторых синапсов

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .weightInit(WeightInit.DISTRIBUTION)
            .dist(new NormalDistribution(0.0, 0.01))
            .activation(Activation.RELU)
            .updater(Updater.NESTEROVS)
            .iterations(iterations)
            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(1e-2)
            .biasLearningRate(1e-2*2)
            .learningRateDecayPolicy(LearningRatePolicy.Step)
            .lrPolicyDecayRate(0.1)
            .lrPolicySteps(100000)
            .regularization(true)
            .l2(5 * 1e-4)
            .momentum(0.9)
            .miniBatch(false)
            .list()
            .layer(0, convInit("cnn1", channels, 96, new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0))
            .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
            .layer(2, maxPool("maxpool1", new int[]{3,3}))
            .layer(3, conv5x5("cnn2", 256, new int[] {1,1}, new int[] {2,2}, nonZeroBias))
            .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
            .layer(5, maxPool("maxpool2", new int[]{3,3}))
            .layer(6,conv3x3("cnn3", 384, 0))
            .layer(7,conv3x3("cnn4", 384, nonZeroBias))
            .layer(8,conv3x3("cnn5", 256, nonZeroBias))
            .layer(9, maxPool("maxpool3", new int[]{3,3}))
            .layer(10, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
            .layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
            .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .name("output")
                .nOut(numLabels)
                .activation(Activation.SOFTMAX)
                .build())
            .backprop(true)
            .pretrain(false)
            .setInputType(InputType.convolutional(height, width, channels))
            .build();

        return new MultiLayerNetwork(conf);

    }

    public static void main(String[] args) throws Exception {
        new ObjectsClassification().run(args);
    }

}
