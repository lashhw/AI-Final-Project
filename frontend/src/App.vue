<script>
import LineChart from './components/LineChart.vue'

export default {
  data() {
    return {
      id: null,
      result: {}
    }
  },
  computed: {
    chartData() {
      return {
        labels: Object.keys(this.result),
        datasets: [ {
            label: null,
            data: Object.values(this.result) 
          }
        ]
      }
    }
  },
  methods: {
    getPrediction(event) {
      fetch(
        `https://ntpcparking.azurewebsites.net/api/predictfuture?id=${this.id}`
      ).then(res => {
        return res.json()
      }).then(result => {
        this.result = result
      })
    }
  },
  components: { LineChart }
}
</script>

<template>
  <input v-model="id" placeholder="停車場 ID" />
  <button @click.prevent="getPrediction"> 送出 </button>
  <LineChart :chart-data='chartData' />
</template>

<style>

</style>
