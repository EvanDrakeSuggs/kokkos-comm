// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <stream-triggering.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include "request.hpp"
#include "stream_space.hpp"

namespace KokkosComm::Experimental {
  namespace stream {
  
template <typename CommSpace = DefaultCommunicationSpace>
class StreamChannel {
 public:
  explicit StreamChannel(int dest_rank, int src_rank, int tag, MPI_Comm comm)
      : dest_rank_(dest_rank), src_rank_(src_rank), tag_(tag), comm_(comm) {
    // set up mem_info, etc.
    // possible switch between fine and coarse
      if ( const char* env_db =
	   std::getenv( "MPI_ADVANCE_DOUBLE_BUFFERING" ) )
	{
	  double_buffer_ = atoi( env_db );
	}
      // if no env variable is found, set to zero
      else
	{
	  double_buffer_ = 1;
	}
      // set up fine grain memory
      if ( const char* env_fg =
	   std::getenv( "MPI_ADVANCE_FINEGRAIN_MEMORY" ) )
	{
	  fine_grain_ = atoi( env_fg );
	}
      // if no env variable is found, set to zero
      else
	{
	  fine_grain_ = 0;
	}
      MPI_Info_create( &mem_info_ );
      if ( fine_grain_ )
        {
	  MPI_Info_set( mem_info_, "mpi_memory_alloc_kinds",
			"rocm:device:fine" );
        }
      else
        {
	  MPI_Info_set( mem_info_, "mpi_memory_alloc_kinds",
			"rocm:device:coarse" );
        }

      if constexpr ( std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::HIP>){
		 my_stream_ = Kokkos::HIP().hip_stream();
  }
      else{
	my_stream_ = nullptr;
      }
      MPIS_Queue_init(&my_queue_, CXI, &my_stream_);
  }

  ~StreamChannel()
  {
    // free send request
    for(int i = 0; i < send_reqs_.size(); i++){
      MPIS_Request_free(&send_reqs_[i]);
    }
    // free recv requests
    for(int i = 0; i < recv_reqs_.size(); i++){
      MPIS_Request_free(&recv_reqs_[i]);
    }
    // free queue
    if(my_queue_ != MPIS_QUEUE_NULL){
      MPIS_Queue_free( &my_queue_ );
    }
    if(mem_info_ != MPI_INFO_NULL){
      MPI_Info_free( &mem_info_ );
    }
  }


  // No copying
  StreamChannel (const StreamChannel& other) = delete;
  StreamChannel& operator=(const StreamChannel& other) = delete;

  // only moving
  StreamChannel(StreamChannel&& other) noexcept
    : //send_reqs_(other.send_reqs_),
    send_reqs_(std::move(other.send_reqs_)),
    //recv_reqs_(other.recv_reqs_),
    recv_reqs_(std::move(other.recv_reqs_)),
    dest_rank_(other.dest_rank_),
    src_rank_(other.src_rank_),
    tag_(other.tag_),
    comm_(other.comm_),
    my_stream_(other.my_stream_),
    my_queue_(other.my_queue_),
    mem_info_(other.mem_info_),
    double_buffer_(other.double_buffer_),
    fine_grain_(other.fine_grain_)
  {
    // clear other
    other.send_reqs_.clear();
    other.recv_reqs_.clear();
    other.dest_rank_ = 0;
    other.src_rank_ = 0;
    other.tag_ = 0;
    other.comm_ = MPI_COMM_NULL;
    other.my_stream_ = nullptr;
    other.mem_info_ = MPI_INFO_NULL;
    other.my_queue_ = MPIS_QUEUE_NULL;
    other.double_buffer_ = 0;
    other.fine_grain_ = 0;
  }

  StreamChannel& operator=(StreamChannel&& other) noexcept
  {
    if (this !=&other){
      send_reqs_ = std::move(other.send_reqs_);
      recv_reqs_ = std::move(other.recv_reqs_);
      dest_rank_ = other.dest_rank_;
      src_rank_ = other.src_rank_;
      tag_ = other.tag_;
      comm_ = other.comm_;
      my_stream_ = other.my_stream_;
      mem_info_ = other.mem_info_;
      my_queue_ = other.my_queue_;
      double_buffer_ = other.double_buffer_;
      fine_grain_ = other.fine_grain_;
      //clear other
      other.send_reqs_.clear();
      other.recv_reqs_.clear();
      other.dest_rank_ = 0;
      other.src_rank_ = 0;
      other.tag_ = 0;
      other.comm_ = MPI_COMM_NULL;
      other.my_stream_ = nullptr;
      other.mem_info_ = MPI_INFO_NULL;
      other.my_queue_ = MPIS_QUEUE_NULL;
      other.double_buffer_ = 0;
      other.fine_grain_ = 0;
    }
    return *this;
      
    }
  
  // Send initialization - dynamically adds a send request to the send queue
  // TODO make send and recv return request object!
  template <class SendView>
  auto sendinit(SendView view) {
    Kokkos::Tools::pushRegion("KokkosComm::StreamChannel::sendinit");
    using value_type = typename SendView::value_type;
    // Add a new request to the send_reqs_ vector
    if( send_reqs_.empty()){
      send_reqs_.emplace_back();
    }
    MPIS_Send_init(
        KokkosComm::data_handle(view), KokkosComm::span(view), datatype<StreamSpace, value_type>(), dest_rank_, tag_,
        comm_, mem_info_,
	&send_reqs_.back()
    );
    MPIS_Match(&send_reqs_.back(), MPI_STATUS_IGNORE);
    Kokkos::Tools::popRegion();
    return &send_reqs_.back();
  }

  // Receive initialization - dynamically adds a receive request to the recv queue
  template <class RecvView>
  auto recvinit(RecvView view) {
    Kokkos::Tools::pushRegion("KokkosComm::StreamChannel::recvinit");
    using value_type = typename RecvView::value_type;
    if( recv_reqs_.empty() ){
      recv_reqs_.emplace_back();
    }
    MPIS_Recv_init(
        KokkosComm::data_handle(view), KokkosComm::span(view), datatype<StreamSpace, value_type>(), src_rank_, tag_, comm_,
	mem_info_,
	&recv_reqs_.back()
    );
    MPIS_Match(&recv_reqs_.back(), MPI_STATUS_IGNORE);
    Kokkos::Tools::popRegion();
    return &recv_reqs_.back();
  }

  // make a start that works on specific requests
  int start( MPIS_Request* my_req ){
    Kokkos::Tools::pushRegion("KokkosComm::StreamChannel::start");
    MPIS_Enqueue_start( my_queue_, my_req);
    Kokkos::Tools::popRegion();
    return MPIS_SUCCESS;
  }

  // make a start that works on specific requests
  int start( int len, MPIS_Request my_req[] ){
    Kokkos::Tools::pushRegion("KokkosComm::StreamChannel::start");
    for(int i = 0; i < len; ++i){
      MPIS_Enqueue_start( my_queue_, &my_req[i]);
    }
    Kokkos::Tools::popRegion();
    return MPIS_SUCCESS;
  } 

  int startall() {
    Kokkos::Tools::pushRegion("KokkosComm::StreamChannel::startall");
    std::vector<MPIS_Request> mpi_reqs;
    /*for (auto& req : send_reqs_) {
      mpi_reqs.push_back(req.request());
    }
    for (auto& req : recv_reqs_) {
      mpi_reqs.push_back(req.request());
    }*/
    MPIS_Enqueue_startall( my_queue_, recv_reqs_.size(), &recv_reqs_[0]);
    MPIS_Enqueue_startall( my_queue_, send_reqs_.size(), &send_reqs_[0]);
    Kokkos::Tools::popRegion();
    return MPIS_SUCCESS;
  }

  // make a wait that works on specific requests
  int wait( MPIS_Request* my_req){
    Kokkos::Tools::pushRegion("KokkosComm::StreamChannel::wait");
    MPIS_Wait(my_req, MPI_STATUS_IGNORE);
    MPIS_Queue_wait( my_queue_ );
    Kokkos::Tools::popRegion();
    return MPIS_SUCCESS;
  }

  int wait( int len, MPIS_Request my_req[]){
    Kokkos::Tools::pushRegion("KokkosComm::StreamChannel::wait");
    for(int i = 0; i < len; ++i){
      MPIS_Wait(&my_req[i], MPI_STATUS_IGNORE);
    }
    MPIS_Queue_wait( my_queue_ );
    Kokkos::Tools::popRegion();
    return MPIS_SUCCESS;
  }  
  
  int waitall() {
    Kokkos::Tools::pushRegion("KokkosComm::StreamChannel::waitall");
    //std::vector<Request<StreamSpace>> reqs;
    //reqs.reserve(send_reqs_.size() + recv_reqs_.size());
    //reqs.insert(reqs.end(), std::make_move_iterator(send_reqs_.begin()), std::make_move_iterator(send_reqs_.end()));
    //reqs.insert(reqs.end(), std::make_move_iterator(recv_reqs_.begin()), std::make_move_iterator(recv_reqs_.end()));
    MPIS_Enqueue_waitall( my_queue_);
    MPIS_Queue_wait( my_queue_ );
    Kokkos::Tools::popRegion();
    return MPIS_SUCCESS;
  }

 private:
  //std::vector<Request<StreamSpace>> send_reqs_;  // Queue for send requests
  //std::vector<Request<StreamSpace>> recv_reqs_;  // Queue for receive requests
  std::vector<MPIS_Request> send_reqs_;  // Queue for send requests
  std::vector<MPIS_Request> recv_reqs_;  // Queue for receive requests
  int dest_rank_;                             // Destination rank for send
  int src_rank_;                              // Source rank for receive
  int tag_;                                   // MPI tag
  MPI_Comm comm_;                             // MPI communicator
  // MPIS stream-triggering items
  void* my_stream_;
  MPIS_Queue my_queue_;
  MPI_Info mem_info_;
  // not implemented yet
  int double_buffer_, fine_grain_;
};

  } // namespace stream
}  // namespace KokkosComm
