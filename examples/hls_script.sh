PROJECT_DIR=`pwd`
BIN=`realpath ./$1`
KERNEL_DIR=`realpath ./$2`

echo "$BIN"
echo "$KERNEL_DIR"

cd "$PROJECT_DIR/build"
mkdir -p "$KERNEL_DIR/tor-raw"
mkdir -p "$KERNEL_DIR/tor-split"
mkdir -p "$KERNEL_DIR/hec"
mkdir -p "$KERNEL_DIR/chisel"
mkdir -p "$KERNEL_DIR/hec_dyn"
mkdir -p "$KERNEL_DIR/chisel_dyn"

#KERNELNAMES=`ls $KERNEL_DIR/tuned/*.mlir`
SCF_TO_TOR=--scf-to-tor
SCHEDULE=--schedule-tor
SPLIT=--split-schedule
HECGEN=--generate-hec
DYNAMIC=--dynamic-schedule
DUMPCHISEL=--dump-chisel

echo "Clean old results"
rm -f $KERNEL_DIR/tor-raw/*
rm -f $KERNEL_DIR/tor-split/*
rm -f $KERNEL_DIR/hec/*
rm -f $KERNEL_DIR/chisel/*
rm -f $KERNEL_DIR/hec_dyn/*
rm -f $KERNEL_DIR/chisel_dyn/*

for kernel_dir in "$KERNEL_DIR"/tuned/*.mlir
do
  kernel="$(basename "$kernel_dir")"
  chisel_file="${kernel%.*}.scala"
  fail_log="${kernel%.*}.log"
  echo "kernel: $kernel"

  COMMAND0="$BIN $SCF_TO_TOR $SCHEDULE "$KERNEL_DIR/tuned/$kernel" -o "$KERNEL_DIR/tor-raw/$kernel""
  echo "$COMMAND0"
  $BIN $SCF_TO_TOR $SCHEDULE "$KERNEL_DIR/tuned/$kernel" -o "$KERNEL_DIR/tor-raw/$kernel" >out 2>error

  COMMAND1="$BIN $SPLIT "$KERNEL_DIR/tor-raw/$kernel" -o "$KERNEL_DIR/tor-split/$kernel""
  echo "$COMMAND1"
  $BIN $SPLIT "$KERNEL_DIR/tor-raw/$kernel" -o "$KERNEL_DIR/tor-split/$kernel" >out 2>error

  COMMAND2="$BIN $HECGEN "$KERNEL_DIR/tor-split/$kernel" -o "$KERNEL_DIR/hec/$kernel""
  echo "$COMMAND2"
  $BIN $HECGEN "$KERNEL_DIR/tor-split/$kernel" -o "$KERNEL_DIR/hec/$kernel" >out 2>error
  
  COMMAND3="$BIN $DYNAMIC "$KERNEL_DIR/tor-raw/$kernel" -o "$KERNEL_DIR/hec_dyn/$kernel""
  echo "$COMMAND3"
  $BIN $DYNAMIC -mlir-disable-threading "$KERNEL_DIR/tor-raw/$kernel" -o "$KERNEL_DIR/hec_dyn/$kernel" >out 2>error
  
  $BIN $SCF_TO_TOR $SCHEDULE $SPLIT $HECGEN $KERNEL_DIR/tuned/$kernel -o $KERNEL_DIR/hec/$kernel >out 2>error
done

for kernel_dir in "$KERNEL_DIR"/hec/*.mlir
do
  kernel="$(basename "$kernel_dir")"
  chisel_file="${kernel%.*}.scala"
  fail_log="${kernel%.*}.log"
  echo "kernel: $kernel"

  COMMAND4="$BIN $DUMPCHISEL "$KERNEL_DIR/hec/$kernel" > "$KERNEL_DIR/chisel/$chisel_file""
  echo "$COMMAND4"
  $BIN $DUMPCHISEL "$KERNEL_DIR/hec/$kernel" >"$KERNEL_DIR/chisel/$chisel_file" 2>error
done

for kernel_dir in "$KERNEL_DIR"/hec_dyn/*.mlir
do
  kernel="$(basename "$kernel_dir")"
  chisel_file="${kernel%.*}.scala"
  fail_log="${kernel%.*}.log"
  echo "kernel: $kernel"

  COMMAND5="$BIN $DUMPCHISEL "$KERNEL_DIR/hec_dyn/$kernel" > "$KERNEL_DIR/chisel_dyn/$chisel_file""
  echo "$COMMAND5"
  $BIN $DUMPCHISEL "$KERNEL_DIR/hec_dyn/$kernel" >"$KERNEL_DIR/chisel_dyn/$chisel_file" 2>error
done
