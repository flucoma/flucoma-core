#include <stdio.h>
#include <Eigen/Dense>
#include "data/FluidTensor.hpp"
#include "util/audiofile.hpp"

using fluid::audiofile::AudioFileData;
using fluid::audiofile::readFile;

using  fluid::FluidTensor;
using  fluid::FluidTensorView;
using fluid::slice;


//void whynoworky(const FluidTensorView<double, 1>& v)
//{
//  double x = *std::max_element(v.begin(), v.end());
//}

void whynoworky(const std::vector<double>& v)
{
  double x = *std::max_element(v.begin(), v.end());
}


int main(int argc, char* argv[])
{
  //Test wrapping interleaved structure, coz I keep getting it wrong
  //4 channels 3 frames
  std::vector<double> quad = {0,1,2,3,0,1,2,3,0,1,2,3};
  
  
  FluidTensorView<double,2> quadview(quad.data(),0,3,4);
  
  std::cout<< quadview.col(0) << '\n';
  
  
  
  
  
  //Wrap any old pointer
    std::vector<double> s = {0,1,2,3,4,5,6,7,8};
  
    whynoworky(s);
  
  
//    FluidTensorView<int,2> s_wrap(s.data(),0,9u,1u);
  
  FluidTensor<double, 2> iteratorWeirdness(9,1);
  auto b = iteratorWeirdness(slice(0),slice(0));
  std::iota(b.begin(),b.end(),0);
  
  
  
//  whynoworky(iteratorWeirdness);
  std::cout<< *std::max_element(b.begin(), b.end()) <<'\n';
  
//    std::cout << s_wrap << '\n';
  
    //zero size, nullptr test
    FluidTensorView<int,2> wrap_null(nullptr,0,0u,0u);
    
  
  FluidTensor<double, 1> onedinit{{1.,2.,34.,5.,6.,7.}}; 
  
  
    FluidTensor<int, 2> threebythree{{0,1,2},{3,4,5},{6,7,8}};
    
    auto col1 = threebythree(slice(0),slice(1,1)); //all the rows, first column
    
    auto twobytwo = threebythree(slice(1,2),slice(1,2));
    
    auto threebytwo = threebythree(slice(0,3),slice(1,2));
    
    std::cout << threebytwo ;
    
    int j = 0;
    for(auto&& i: threebytwo)
    {
        std::cout << j++ << ' '  << i << '\n';
    }
    
    return 0;
    
    //We can initialize a tensor with some elements using braces:
    fluid::FluidTensor<double,1> tinit{1.0,2.0,3.0};
    std::cout << "Rank: " << tinit.order << " Length: "
    << tinit.extent(0) << " Data: "<< tinit << '\n';

    //We can nest them for multiple dimenions
    fluid::FluidTensor<double,2> tinit2{{1.0,2.0,3.0},{4,5,6}};
    std::cout << "Rank : " << tinit2.order << " Dimension 1: " << tinit2.extent(0)<<" Dimension 2: "<<  tinit2.extent(1) << " Data: "<< tinit2 << '\n';

    //Three dimensions....
    fluid::FluidTensor<double, 3> threedee{
        {{1,2},{3,4}}, {{5,6},{7,8}}
    };
    std::cout << threedee << '\n';

    //We can copy tensors
    fluid::FluidTensor<double,2> tcopy3 = tinit2;
    std::cout << tcopy3.order << " " << tcopy3.extent(0)<<" "<<
    tcopy3.extent(1) << '\n';

    //Grab a row:
    auto r1 = tinit2.row(1);
    std::cout << "4 5 6?" << r1 << '\n';


    fluid::FluidTensor<double,1> r2(tinit2.row(1));
    std::cout << "1 2 3?" << r1 << '\n';


    //tinit2.row(1) = r2(fluid::slice(0,3));
    //Initialize with vector
    std::cout << "tinit2"<<tinit2 << '\n';
    std::vector<double> v{1,2,3};
    fluid::FluidTensor<double,1> bob(v);
    std::cout << bob << '\n';

    //Initialize with dimenions
    fluid::FluidTensor<double, 2> ermintrude(3,3);

    //Eigen mapping.
    using FluidTensorToEigen = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

    //This is an Eigen::Map:
    FluidTensorToEigen maptoflu(tinit2.data(),tinit2.extent(0),tinit2.extent(1));

    //Multiply everything by 3 (with Eigen)
    maptoflu *= 3;
    //Print with Eigen
    std::cout << "maptoflu"<<maptoflu << '\n';
    //Print with FluidTensor
    std::cout << "tinit2"<<tinit2 << '\n';

    std::cout << "tinit2 raw"<<tinit2.data()[3] << '\n';



    //Map a row
    using FluidTensorToEigenVector = Eigen::Map<Eigen::Matrix<double,  1,  Eigen::Dynamic,Eigen::RowMajor>>;

    //This is also an eigen map, just looking at a row
    FluidTensorToEigenVector maptorow(r1.data(),r1.rows());

    //Divide that row back to original
    maptorow /= 3;
    //Print with Eigen
    std::cout << maptorow << '\n';
    //This is what we expect:
    std::array<size_t,3> r1_expected{{4,5,6}};
    //Is it true?
    assert(std::equal(r1_expected.begin(), r1_expected.end(),r1.data()));
    //Print with FluidTensor
    std::cout << r1 << '\n';

    //Grab a column
    auto c1 = tinit2.col(2);
    //Have a look
    std::cout << c1 << '\n';
    //Should have this in it:
    std::array<size_t,6> tinit2_expected{{3,6,9,4,5,6}};
    //Let's look
    std::cout << tinit2 << '\n';
    //Is it right?
    assert(std::equal(tinit2_expected.begin(), tinit2_expected.end(),tinit2.data()));

    //Test double** constructor for 2D
    size_t x(10);
    size_t y(10);
    //Make a pointer to double[]
    double* twodeecee[x];
    for(int i = 0; i < x; ++i)
    {
        twodeecee[i] = new double[y];
        std::iota(twodeecee[i], twodeecee[i] + y, i * y);
    }
    //Create
    fluid::FluidTensor<double, 2> twodee_test(twodeecee,x,y);
    //Look
    std::cout << twodee_test << '\n';

    //Test stepping through a column whilst we're here
    size_t col_offset = 3;
    auto c2 = twodee_test.col(col_offset);
    for(int i = 0; i < y; i++)
    {
        //use() with integer types to get elements
        assert(c2[i] == twodeecee[i][col_offset]);
    }
    std::cout << "Col 3: " << c2 << '\n';

    //Free memory from double**, we're done with it
    for(int i = 0; i < x; ++i)
        delete twodeecee[i];

    //Make a new blank 2D tensor
    fluid::FluidTensor<double,2> copy_col(2,10);

    //Use() operator with at least one fluid::slice(start,size,stride), possibly mixed
    //with integer types to do arbitary slicees:

    //Take three values from column above, with a stride of 2, offset of 0 and
    //copy to to the begning of the first row of our blank tensor
    copy_col.row(0)(fluid::slice(0,3)) = c2(fluid::slice(0,3,2));

    fluid::FluidTensorView<double,1> aa = c2(fluid::slice(0,3,2));

    std::cout<<"Original column " << c2 <<'\n';

    std::cout<<"Sub-column " << aa[0] <<'\n';

    std::cout<<"Copied to row " << copy_col <<'\n';


    //More arbitary slicing
    size_t offset = 5;
    size_t len = 5;
    auto sli = twodee_test(3,fluid::slice(offset, len));
    std::cout << " Slice, start at 5, length 5, start from row 3 " << sli << '\n';
    //Slice from row zero, column 3, lnegth 3 => 10 x 3
    auto sli2 = twodee_test(fluid::slice(0),fluid::slice(3, 3));
    std::cout << "Slice from row zero, column 3, length 3 => 10 x 3\n" << sli2 << '\n';
    //Slice from row zero to 3, column 3, length 3 => 3 x 3
    auto sli3 = twodee_test(fluid::slice(0, 3),fluid::slice(3, 3));
    std::cout << "Slice from row zero to 3, column 3, length 3 => 3 x 3\n"<< sli3 << '\n';


    //Test a bigger vector for 1D
    AudioFileData data = readFile("/Applications/Max.app/Contents/Resources/C74/media/msp/jongly.aif");
    std::vector<std::vector<double>> a = data.audio;
    //Init from vector
    fluid::FluidTensor<double, 1> audio_test(data.audio[0]);
    //Sizes should match
    assert(audio_test.size() == data.audio[0].size());
    //Data should match
    assert(std::equal(data.audio[0].begin(), data.audio[0].end(),
                      audio_test.data()));
    //Data should also match when accessed through operator()!
    for(int i = 0; i < data.audio[0].size();++i)
    {
        assert(audio_test[i] == data.audio[0][i]);
    }

    //Test unary apply
    fluid::FluidTensor<double,1> seq(10);
    std::iota(seq.begin(),seq.end(),0);

    std::cout<<"sequence \n" << seq << '\n';
    seq.apply([](double & x){x *= 2;});
    std::cout<<"sequence doubled\n" << seq << '\n';



    //Test binary apply
    fluid::FluidTensor<double,2> sincos(3,1024);
    fluid::FluidTensor<double,1> ramp(1024);
    std::iota(ramp.begin(),ramp.end(),0);
    auto sin = sincos.row(0);
    auto cos = sincos.row(1);
    auto sqr = sincos.row(2);

    sin.apply(ramp,[](double& x, double idx){
        x = std::sin(M_2_PI * idx * 32. * (1024. / 44100.)); //some arbitary sine wave
    });

    cos.apply(ramp,[](double& x, double idx){
        x = std::cos(M_2_PI * idx * 32. * (1024. / 44100.));//some arbitary cosine wave
    });
//    //Hmm, now I want I variadic version. Oh well
    sqr.apply(sin,[](double& x, double& y){x=y;});
    sqr.apply(cos,[](double& x, double& y){x=(x*x) + (y*y);});
//
    std::cout << "sin^2 + cos^2 " << sqr <<'\n'; //ought to be 1ish

   //Can we overlap-add?
    size_t hop = 256;
    size_t frame = 1024;
    fluid::FluidTensor<double,1> in(1024 * 100);
    fluid::FluidTensor<double,1> out(1024 * 100);
    //Leave frame/2 at either end
    auto in_minus_padding = in(fluid::slice(frame/2, frame*99));
    //Zero output buffer, just in case
    out.apply([](double& x){x=0;});

    //Make a window
    fluid::FluidTensor<double,1> win(1024);
    std::iota(win.begin(),win.end(),0);//make ramp
    win.apply([](double& x){
        x = 0.5 - 0.5*(std::cos((M_PI * 2 * x) / 1024)); //periodic hann window (1024,not 1023 in denominator)
    });

    size_t win_sum = std::accumulate(win.begin(), win.end(), 0);
    //normalise window to sum to 1
    win.apply([&](double& x){
        x /= win_sum;
    });

    //Fill input with a cosine
    std::iota(in_minus_padding.begin(), in_minus_padding.end(),0);
    in_minus_padding.apply([](double& x){
        x = std::cos(2 * M_PI * x * 32 * (1024./44100.));
    });

    //Make a normalisation buffer
    fluid::FluidTensor<double,1> norm(out);

    //Loop over frames, advancing by 1 hop
    for(int i = 0; i < 99*frame; i+=hop)
    {
        //Same slice all round: start at i, take frame samples
        fluid::slice slice(i,frame);
        //grab input and copy before windowing
        fluid::FluidTensor<double, 1> windowed(in(slice));
        //apply window
        windowed.apply(win, [](double& x, double w){ x *= w; });
        //accumulate
        out(slice).apply(windowed,[](double& x,double y){ x += y; });
        //accumulate window into normalisation buffer
        norm(slice).apply(win,[](double& x, double y){ x += y; });
    }
    //normalise
    out.apply(norm,[](double& x, double y){
          if(x != 0)
            x /=  y > 0? y : 1 ;
    });
    //take the difference
    fluid::FluidTensor<double, 1> diff(in);
    diff.apply(out, [](double& x, double y){ x -= y; });
    //sum it
    double sum_of_diff = std::accumulate(diff.begin(), diff.end(), 0.0);
    //we want it to be small, kthx
    std::cout << "Difference summed " << sum_of_diff << '\n';

    double* interleave = new double[100];

    std::iota(interleave, interleave+100,0);

    //Test for assumption about reading from interleaved structure (e.g multichannel buffers in max and sc)
    fluid::FluidTensorView<double,2> interT = fluid::FluidTensorView<double,2>({0,{50,2}},interleave);

    std::cout << interT.col(0) << '\n';


}
