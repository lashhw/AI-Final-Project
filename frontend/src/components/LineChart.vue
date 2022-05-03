<template>
  <Line
    :chart-data="chartData"
    :chart-options="chartOptions"
  />
</template>

<script>
import { Line } from 'vue-chartjs'
import Chart from 'chart.js/auto';
import 'chartjs-adapter-date-fns';

export default {
  name: 'BarChart',
  components: { Line },
  props: {  
    chartData: {
      type: Object,
      required: true
    },
    chartOptions: {
      type: Object,
      default: () => {}
    }
  }
}
</script>